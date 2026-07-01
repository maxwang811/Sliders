#!/usr/bin/env python3
"""End-to-end smoke test for the SLIDERS pipeline *and* the evaluation path.

The bundled CLI (``run_sliders``) only runs the answering pipeline; the
LLM-as-a-judge evaluators are exercised by the benchmark drivers. This script
ties both together against the bundled ``sample_docs/`` so you can confirm, in
one shot, that:

  1. the full pipeline runs (chunking -> schema induction -> extraction ->
     reconciliation -> SQL answer synthesis), and
  2. the evaluation path works, using the dedicated evaluator credentials from
     the ``EVAL_*`` environment variables (e.g. ``EVAL_OPENAI_API_KEY``).

The evaluation section runs three checks:
  * a framing-aligned correct answer -> both judges (soft + strict hard) must
    say correct;
  * a deliberately wrong answer -> both judges must say incorrect; and
  * the real (free-form) pipeline answer -> the soft/semantic judge must say
    correct (the hard judge is intentionally strict about exact framing, so its
    verdict on free-form text is reported but not used as a gate).
A passing run proves the evaluator LLM is actually being called and is
discriminating -- not just returning a constant.

Usage:
    uv run python verify_eval_pipeline.py            # full pipeline + evaluation
    uv run python verify_eval_pipeline.py --quick    # evaluation path only (fast)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

# Importing globals initializes the prompt (Jinja) environment and loads the
# repo .env (with override), which is what makes the evaluators usable here.
import sliders.globals  # noqa: F401  (side effects: init_llm + load_dotenv)

from sliders.evaluation import Evaluator, LLMAsJudgeEvaluationTool

QUESTION = "How many participants were randomized in each of the two trials, and which trial randomized more?"

GOLD_ANSWER = (
    "The alphaxolin hypertension trial randomized 420 participants and the betapril HFpEF "
    "trial randomized 284 participants, so the alphaxolin trial randomized more."
)

# Correct paraphrase used in --quick mode (numbers and framing match the gold).
CANNED_CORRECT_ANSWER = (
    "The alphaxolin trial enrolled 420 participants while the betapril trial enrolled 284, "
    "so alphaxolin had the larger sample size."
)

# Deliberately wrong answer: numbers and conclusion are swapped.
WRONG_ANSWER = (
    "The betapril trial randomized 420 participants and the alphaxolin trial randomized 284, "
    "so the betapril trial randomized more."
)

EVALUATOR_PROMPTS = (
    "evaluators/soft_evaluator.prompt",
    "evaluators/hard_evaluator.prompt",
)
EVALUATOR_MODEL = "gpt-4.1"


def _fingerprint(value: str | None) -> str:
    if not value:
        return "<unset>"
    return f"...{value[-6:]} (len={len(value)})"


def print_eval_config() -> None:
    print("=" * 78)
    print("EVALUATOR LLM CONFIGURATION (read from environment / .env)")
    print("=" * 78)
    print(f"  EVAL_LLM_PROVIDER   : {os.getenv('EVAL_LLM_PROVIDER') or '<unset>'}")
    print(f"  EVAL_OPENAI_API_KEY : {_fingerprint(os.getenv('EVAL_OPENAI_API_KEY'))}")
    print(f"  EVAL_OPENAI_BASE_URL: {os.getenv('EVAL_OPENAI_BASE_URL') or '<default: https://api.openai.com/v1>'}")
    print(f"  EVAL_AZURE_ENDPOINT : {os.getenv('EVAL_AZURE_OPENAI_ENDPOINT') or '<unset>'}")
    print(f"  evaluator model     : {EVALUATOR_MODEL}")
    print(
        "  (main pipeline key  : "
        f"OPENAI_API_KEY {_fingerprint(os.getenv('OPENAI_API_KEY'))}, "
        f"provider={os.getenv('SLIDERS_LLM_PROVIDER') or '<default: azure>'})"
    )
    print()


def build_evaluator() -> Evaluator:
    """Build the same soft + hard LLM-as-judge evaluator the benchmarks use."""
    evaluator = Evaluator()
    for prompt_file in EVALUATOR_PROMPTS:
        evaluator.add_evaluation_tool(
            LLMAsJudgeEvaluationTool(
                prompt_file=prompt_file,
                model=EVALUATOR_MODEL,
                temperature=0.0,
                max_tokens=4096,
            )
        )
    return evaluator


async def _evaluate(evaluator: Evaluator, predicted: str, question_id: str) -> dict:
    return await evaluator.evaluate(
        question_id=question_id,
        question=QUESTION,
        gold_answer=GOLD_ANSWER,
        predicted_answer=predicted,
    )


def _summarize_tools(label: str, result: dict) -> tuple[dict[str, bool], bool]:
    """Print per-tool verdicts; return {tool: correct_bool} and an error flag."""
    print(f"--- {label} ---")
    tools = result.get("evaluation_tools", {})
    verdicts: dict[str, bool] = {}
    had_error = False
    for tool_name, tool_result in tools.items():
        if "error" in tool_result:
            had_error = True
            print(f"  [{tool_name}] ERROR: {tool_result['error']}")
            continue
        correct = bool(tool_result.get("correct"))
        verdicts[tool_name] = correct
        explanation = (tool_result.get("explanation") or "").strip().replace("\n", " ")
        if len(explanation) > 160:
            explanation = explanation[:157] + "..."
        print(f"  [{tool_name}] correct={correct} | {explanation}")
    print()
    return verdicts, had_error


SOFT_EVALUATOR_KEY = "LLMAsJudgeEvaluationTool_soft_evaluator"


def _gate(label: str, result: dict, *, expect_correct: bool, soft_only: bool = False) -> bool:
    """Print per-tool verdicts and decide whether this check passed.

    expect_correct=True  -> verdict(s) must be correct
    expect_correct=False -> verdict(s) must be incorrect
    soft_only=True       -> gate on the soft/semantic judge only (for free-form text)
    """
    verdicts, had_error = _summarize_tools(label, result)
    if had_error or not verdicts:
        return False
    if expect_correct:
        return verdicts.get(SOFT_EVALUATOR_KEY, False) if soft_only else all(verdicts.values())
    return not any(verdicts.values())


def run_evaluation(pipeline_answer: str | None) -> bool:
    """Exercise the evaluators; return True if every check passes."""
    evaluator = build_evaluator()

    print("=" * 78)
    print("EVALUATION (LLM-as-a-judge)")
    print("=" * 78)
    print(f"Question   : {QUESTION}")
    print(f"Gold answer: {GOLD_ANSWER}")
    print()

    checks: dict[str, bool] = {}

    correct_result = asyncio.run(_evaluate(evaluator, CANNED_CORRECT_ANSWER, "verify_correct"))
    checks["both judges accept a matching answer"] = _gate(
        "Controlled CORRECT answer (expect BOTH judges correct=True)",
        correct_result,
        expect_correct=True,
    )

    wrong_result = asyncio.run(_evaluate(evaluator, WRONG_ANSWER, "verify_wrong"))
    checks["both judges reject a wrong answer"] = _gate(
        "Controlled WRONG answer (expect BOTH judges correct=False)",
        wrong_result,
        expect_correct=False,
    )

    if pipeline_answer is not None:
        pipeline_result = asyncio.run(_evaluate(evaluator, pipeline_answer, "verify_pipeline"))
        checks["soft judge accepts the real pipeline answer"] = _gate(
            "Real PIPELINE answer (expect SOFT/semantic judge correct=True)",
            pipeline_result,
            expect_correct=True,
            soft_only=True,
        )

    print("-" * 78)
    print("CHECKS:")
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print()
    return all(checks.values())


def run_full_pipeline() -> str:
    """Run the full SLIDERS pipeline on sample_docs and return the answer."""
    from sliders.run import run_sliders

    print("=" * 78)
    print("FULL PIPELINE (sample_docs/) - chunking -> schema -> extraction -> "
          "reconciliation -> answer")
    print("=" * 78)
    print(f"Question: {QUESTION}\n", flush=True)

    result = run_sliders(
        docs="sample_docs/",
        question=QUESTION,
        return_full_result=True,
        output_dir="sliders_output/eval_verify",
    )
    answer = result["answer"]
    print("Predicted answer:\n")
    print(answer)
    print()
    print(f"(Full results JSON: {result['results_json_path']})\n")
    return answer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip the (slow) pipeline run and evaluate a canned answer instead.",
    )
    args = parser.parse_args()

    print_eval_config()

    if not os.getenv("EVAL_OPENAI_API_KEY") and not os.getenv("EVAL_AZURE_OPENAI_API_KEY"):
        print(
            "WARNING: no EVAL_OPENAI_API_KEY / EVAL_AZURE_OPENAI_API_KEY set; the evaluator "
            "will fall back to the main pipeline credentials.\n"
        )

    if args.quick:
        print("Running in --quick mode: skipping pipeline; testing evaluator discrimination only.\n")
        ok = run_evaluation(pipeline_answer=None)
    else:
        predicted = run_full_pipeline()
        ok = run_evaluation(pipeline_answer=predicted)

    print()
    print("=" * 78)
    print("OVERALL:", "PASS - full pipeline + evaluation are working." if ok else "FAIL - see messages above.")
    print("=" * 78)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
