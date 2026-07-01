import os
from typing import Callable

import pandas as pd
from tqdm import tqdm
from tqdm.asyncio import tqdm as tqdm_asyncio

from sliders.document import contextualize_document_metadata
from sliders.baselines import System
from sliders.datasets import Dataset
from sliders.document import Document
from sliders.evaluation import Evaluator, LLMAsJudgeEvaluationTool
from sliders.globals import SlidersGlobal
from sliders.log_utils import logger
from sliders.experiments.base import Experiment


# region agent log
import json as _dbg_json
import time as _dbg_time

_DBG_LOG_PATH = "/Users/mohanwang/Desktop/Internships/AI Augmented Database/sliders/.cursor/debug-3303b8.log"


def _dbg(location, message, data, hypothesis_id, run_id="initial"):
    try:
        with open(_DBG_LOG_PATH, "a") as _f:
            _f.write(
                _dbg_json.dumps(
                    {
                        "sessionId": "3303b8",
                        "timestamp": int(_dbg_time.time() * 1000),
                        "location": location,
                        "message": message,
                        "data": data,
                        "hypothesisId": hypothesis_id,
                        "runId": run_id,
                    }
                )
                + "\n"
            )
    except Exception:
        pass
# endregion


def log_finance_bench_results(result):
    logger.info(f"Gold Answer: {result['gold_answer']}")
    logger.info(f"Predicted Answer: {result['predicted_answer']}")
    for key, value in result["evaluation_tools"].items():
        if isinstance(value, dict):
            if "correct" in value:
                logger.info(f"{key}: {value['correct']}")
            else:
                logger.info(f"{key}: {value}")
        else:
            logger.info(f"{key}: {value}")


class FinanceBench(Experiment):
    """Driver for the FinanceBench (Islam et al., 2023) single-document QA benchmark.

    Loads the FinanceBench JSONL question set plus the accompanying markdown
    filings directory, and scores each prediction with two LLM-as-a-judge
    evaluators (``soft`` and ``hard``).
    """

    def __init__(self, config: dict):
        self.config = config

        benchmark_path = self.config.get("benchmark_path")
        files_dir = self.config.get("files_dir")
        gpt_results_path = self.config.get("gpt_results_path")

        if benchmark_path is None:
            benchmark_path = "/path/to/datasets/financebench/data/financebench_open_source.jsonl"
        if files_dir is None:
            files_dir = "/path/to/datasets/financebench/markdown/pdfs/"
        if gpt_results_path is None:
            gpt_results_path = "/path/to/datasets/financebench/results/gpt-4_oracle_reverse.jsonl"

        self.dataset = Dataset(benchmark_path)
        self.dataset = self._apply_filters(self.dataset, config)
        self.gpt_results = pd.read_json(gpt_results_path, lines=True)

        self.files_dir = files_dir
        self.evaluator = Evaluator()

        self.evaluator.add_evaluation_tool(
            LLMAsJudgeEvaluationTool(
                prompt_file="evaluators/soft_evaluator.prompt",
                model=self.config["soft_evaluator_model"],
                temperature=0.0,
                max_tokens=4096,
            )
        )
        self.evaluator.add_evaluation_tool(
            LLMAsJudgeEvaluationTool(
                prompt_file="evaluators/hard_evaluator.prompt",
                model=self.config["hard_evaluator_model"],
                temperature=0.0,
                max_tokens=4096,
            )
        )

    def _apply_filters(self, dataset: Dataset, config: dict) -> Dataset:
        """Restrict ``dataset`` to the rows selected by ``config`` filters."""
        if config.get("specific_ids_csv"):
            try:
                import pandas as pd

                id_sample_df = pd.read_csv(config["specific_ids_csv"], comment="#")
                specific_ids = set(id_sample_df["id"].tolist())

                # FinanceBench uses 'financebench_id' as the ID field, not 'id'
                filtered_data = [item for item in dataset.data if item.get("financebench_id") in specific_ids]

                # Create new dataset with filtered data
                new_dataset = Dataset.__new__(Dataset)
                new_dataset.path = dataset.path
                new_dataset.data = filtered_data

                logger.info(
                    f"Filtered dataset by specific IDs: {len(specific_ids)} IDs requested, {len(filtered_data)} found"
                )
                return new_dataset
            except Exception as e:
                logger.warning(f"Failed to load specific IDs CSV: {e}, using full dataset")

        return dataset

    @property
    def description(self) -> str:
        return "Financial statement for a company (10K, 10Q, 8K, etc.)"

    async def _run_row(self, row: dict, system: System, all_metadata: list) -> dict:
        question = row["question"]
        file_path = os.path.join(self.files_dir, row["doc_name"] + ".md")
        # region agent log
        try:
            _md_exists = os.path.exists(file_path)
            _md_size = os.path.getsize(file_path) if _md_exists else None
            _pdf_dir = os.path.join(os.path.dirname(self.files_dir.rstrip("/")), "pdfs")
            _pdf_path = os.path.join(_pdf_dir, row["doc_name"] + ".pdf")
            _md_count = (
                len([n for n in os.listdir(self.files_dir) if n.endswith(".md")])
                if os.path.isdir(self.files_dir)
                else -1
            )
            _dbg(
                "finance_bench.py:_run_row",
                "markdown load attempt",
                {
                    "doc_name": row.get("doc_name"),
                    "question_id": row.get("financebench_id"),
                    "file_path": file_path,
                    "file_path_abs": os.path.abspath(file_path),
                    "md_exists": _md_exists,
                    "md_size": _md_size,
                    "pdf_exists": os.path.exists(_pdf_path),
                    "files_dir": self.files_dir,
                    "files_dir_exists": os.path.isdir(self.files_dir),
                    "md_count_in_dir": _md_count,
                    "cwd": os.getcwd(),
                },
                hypothesis_id="A,B,C,E",
            )
        except Exception:
            pass
        # endregion
        try:
            # Load + contextualize the document inside the try/except so a single
            # bad/missing markdown file yields a structured error result instead
            # of raising and aborting the whole (parallel) run.
            document = await Document.from_markdown(
                file_path,
                description=self.description,
                document_name=row["doc_name"],
                **self.config.get("document_config", {}),
            )

            logger.info(f"Number of chunks: {len(document.chunks)}")
            if self.config.get("docprocesssing", True):
                all_documents = await contextualize_document_metadata([document], question, model=self.config.get("document_config", {}).get("description_model", "gpt-4.1-mini"))
            else:
                all_documents = [document]

            answer, metadata = await system.run(question, all_documents, question_id=row["financebench_id"])
            metadata["gold_answer"] = row["answer"]
            metadata["predicted_answer"] = answer
        except Exception as e:
            logger.error(f"Error running system for question: {question}")
            import traceback

            logger.error(traceback.format_exc())
            logger.error(e)
            all_metadata.append(
                {
                    "question": question,
                    "error": str(e),
                    "answer": None,
                    "metadata": None,
                    "question_id": row["financebench_id"],
                }
            )
            return {"error": str(e), "question_id": row["financebench_id"]}

        metadata["id"] = row["financebench_id"]
        metadata["evidence"] = row["evidence"]

        # Check if we have both pre-merge and post-merge answers
        if "pre_merge_answer" in metadata and "post_merge_answer" in metadata:
            # Evaluate pre-merge answer
            pre_merge_evaluation = await self.evaluator.evaluate(
                question_id=row["financebench_id"],
                question=question,
                gold_answer=row["answer"],
                predicted_answer=metadata["pre_merge_answer"],
            )

            # Evaluate post-merge answer
            post_merge_evaluation = await self.evaluator.evaluate(
                question_id=row["financebench_id"],
                question=question,
                gold_answer=row["answer"],
                predicted_answer=metadata["post_merge_answer"],
            )

            # Store both evaluations
            result = {
                "question_id": row["financebench_id"],
                "question": question,
                "gold_answer": row["answer"],
                "pre_merge_evaluation": pre_merge_evaluation,
                "post_merge_evaluation": post_merge_evaluation,
            }
        else:
            # Single evaluation
            result = await self.evaluator.evaluate(
                question_id=row["financebench_id"],
                question=question,
                gold_answer=row["answer"],
                predicted_answer=answer,
            )

        all_metadata.append(metadata)
        return result

    async def run(
        self,
        system: System,
        filter_func: Callable[[dict], bool] | None = None,
        sample_size: int | None = None,
        random_state: int | None = None,
        parallel: bool = False,
        **kwargs,
    ) -> dict:
        results = []
        dataset = self.dataset
        if filter_func is not None:
            dataset = dataset.filter(filter_func)
        if sample_size is not None:
            dataset = dataset.sample(sample_size, random_state=random_state)
        all_metadata = []

        dataset_size = len(dataset)

        # region agent log
        try:
            _present, _missing = [], []
            for _row in dataset:
                _fp = os.path.join(self.files_dir, _row["doc_name"] + ".md")
                if os.path.exists(_fp) and os.path.getsize(_fp) > 0:
                    _present.append(_row["doc_name"])
                else:
                    _missing.append(_row["doc_name"])
            _dbg(
                "finance_bench.py:run",
                "preflight markdown availability",
                {
                    "parallel": parallel,
                    "dataset_size": dataset_size,
                    "files_dir": self.files_dir,
                    "files_dir_abs": os.path.abspath(self.files_dir),
                    "files_dir_exists": os.path.isdir(self.files_dir),
                    "cwd": os.getcwd(),
                    "rows_with_md": len(_present),
                    "rows_missing_md": len(_missing),
                    "unique_missing_docs": sorted(set(_missing)),
                },
                hypothesis_id="A,C,D",
            )
        except Exception:
            pass
        # endregion

        if parallel:
            # tqdm_asyncio.gather() forwards unknown kwargs to the tqdm progress
            # bar, so (unlike asyncio.gather) it rejects return_exceptions=True.
            # Normalize failures per-row instead: a single failing row returns
            # its exception rather than aborting the whole gather.
            async def _run_row_safe(row):
                try:
                    return await self._run_row(row, system, all_metadata)
                except Exception as e:  # noqa: BLE001
                    return e

            tasks = [_run_row_safe(row) for row in dataset]
            raw_results = await tqdm_asyncio.gather(*tasks, desc="Evaluating")
            results = []
            for r in raw_results:
                if isinstance(r, BaseException):
                    logger.error(f"Row task failed and was skipped: {r}")
                    results.append({"error": str(r), "question_id": None})
                else:
                    results.append(r)
        else:
            for idx, row in enumerate(tqdm(dataset, desc="Running experiment")):
                logger.info(
                    f"===============================================\n{idx + 1} of {dataset_size} | Question {row.get('financebench_id', 'N/A')}\n==============================================="
                )

                result = await self._run_row(row, system, all_metadata)
                results.append(result)

                # Log both evaluations if we have pre/post-merge
                if "pre_merge_evaluation" in result:
                    logger.info("=" * 80)
                    logger.info("PRE-MERGE EVALUATION:")
                    log_finance_bench_results(result["pre_merge_evaluation"])
                    logger.info("=" * 80)
                    logger.info("POST-MERGE EVALUATION:")
                    log_finance_bench_results(result["post_merge_evaluation"])
                    logger.info("=" * 80)
                elif "error" in result:
                    logger.warning(f"Skipping row with error: {result.get('error')}")
                else:
                    log_finance_bench_results(result)

                # Log current evaluation state and accuracies
                if len(results) > 0 and (results[0].get("evaluation_tools") or results[0].get("pre_merge_evaluation")):
                    # Check if we have pre-merge/post-merge evaluations
                    if "pre_merge_evaluation" in results[0]:
                        # Calculate separate accuracies for pre-merge and post-merge
                        pre_merge_accuracies = {}
                        post_merge_accuracies = {}

                        for eval_tool in results[0]["pre_merge_evaluation"]["evaluation_tools"].keys():
                            # Pre-merge
                            if isinstance(
                                results[0]["pre_merge_evaluation"]["evaluation_tools"][eval_tool]["correct"], bool
                            ):
                                correct_count = sum(
                                    1
                                    for r in results
                                    if r.get("pre_merge_evaluation", {})
                                    .get("evaluation_tools", {})
                                    .get(eval_tool, {})
                                    .get("correct", False)
                                )
                                accuracy = correct_count / len(results)
                                pre_merge_accuracies[eval_tool] = accuracy
                            elif isinstance(
                                results[0]["pre_merge_evaluation"]["evaluation_tools"][eval_tool]["correct"],
                                (int, float),
                            ):
                                correct_count = sum(
                                    r.get("pre_merge_evaluation", {})
                                    .get("evaluation_tools", {})
                                    .get(eval_tool, {})
                                    .get("correct", 0)
                                    for r in results
                                )
                                accuracy = correct_count / len(results)
                                pre_merge_accuracies[eval_tool] = accuracy

                            # Post-merge
                            if isinstance(
                                results[0]["post_merge_evaluation"]["evaluation_tools"][eval_tool]["correct"], bool
                            ):
                                correct_count = sum(
                                    1
                                    for r in results
                                    if r.get("post_merge_evaluation", {})
                                    .get("evaluation_tools", {})
                                    .get(eval_tool, {})
                                    .get("correct", False)
                                )
                                accuracy = correct_count / len(results)
                                post_merge_accuracies[eval_tool] = accuracy
                            elif isinstance(
                                results[0]["post_merge_evaluation"]["evaluation_tools"][eval_tool]["correct"],
                                (int, float),
                            ):
                                correct_count = sum(
                                    r.get("post_merge_evaluation", {})
                                    .get("evaluation_tools", {})
                                    .get(eval_tool, {})
                                    .get("correct", 0)
                                    for r in results
                                )
                                accuracy = correct_count / len(results)
                                post_merge_accuracies[eval_tool] = accuracy

                        logger.info(f"=== CURRENT EVALUATION STATE ({len(results)}/{dataset_size}) ===")
                        logger.info("PRE-MERGE ACCURACIES:")
                        for eval_tool, accuracy in pre_merge_accuracies.items():
                            logger.info(f"  {eval_tool} accuracy: {accuracy:.3f}")
                        logger.info("POST-MERGE ACCURACIES:")
                        for eval_tool, accuracy in post_merge_accuracies.items():
                            logger.info(f"  {eval_tool} accuracy: {accuracy:.3f}")
                    else:
                        # Single evaluation path
                        current_accuracies = {}
                        for eval_tool in results[0]["evaluation_tools"].keys():
                            if isinstance(results[0]["evaluation_tools"][eval_tool]["correct"], bool):
                                correct_count = sum(
                                    1
                                    for r in results
                                    if r.get("evaluation_tools", {}).get(eval_tool, {}).get("correct", False)
                                )
                                accuracy = correct_count / len(results)
                                current_accuracies[eval_tool] = accuracy
                            elif isinstance(results[0]["evaluation_tools"][eval_tool]["correct"], (int, float)):
                                correct_count = sum(
                                    r.get("evaluation_tools", {}).get(eval_tool, {}).get("correct", 0) for r in results
                                )
                                accuracy = correct_count / len(results)
                                current_accuracies[eval_tool] = accuracy

                        logger.info(f"=== CURRENT EVALUATION STATE ({len(results)}/{dataset_size}) ===")
                        for eval_tool, accuracy in current_accuracies.items():
                            logger.info(f"{eval_tool} accuracy: {accuracy:.3f}")

                # Log progress and any errors
                if "error" in result:
                    logger.warning(f"Question {row.get('financebench_id', 'N/A')} had an error: {result['error']}")
                logger.info(f"Completed {len(results)}/{dataset_size} questions")

        for i, result in enumerate(results):
            meta = all_metadata[i] if i < len(all_metadata) else {}
            result["id"] = meta.get("id")
            result["evidence"] = meta.get("evidence")

        # Check if we have pre-merge/post-merge evaluations
        has_split_evaluation = len(results) > 0 and "pre_merge_evaluation" in results[0]

        if has_split_evaluation:
            # Calculate separate summaries for pre-merge and post-merge
            pre_merge_summary = {}
            post_merge_summary = {}

            for result in results:
                if "pre_merge_evaluation" in result:
                    for tool_name, tool_data in result["pre_merge_evaluation"].get("evaluation_tools", {}).items():
                        if tool_name not in pre_merge_summary:
                            pre_merge_summary[tool_name] = {"correct": 0, "total": 0}

                        is_correct = False
                        if isinstance(tool_data, dict):
                            is_correct = bool(tool_data.get("correct", False))

                        pre_merge_summary[tool_name]["correct"] += int(is_correct)
                        pre_merge_summary[tool_name]["total"] += 1

                if "post_merge_evaluation" in result:
                    for tool_name, tool_data in result["post_merge_evaluation"].get("evaluation_tools", {}).items():
                        if tool_name not in post_merge_summary:
                            post_merge_summary[tool_name] = {"correct": 0, "total": 0}

                        is_correct = False
                        if isinstance(tool_data, dict):
                            is_correct = bool(tool_data.get("correct", False))

                        post_merge_summary[tool_name]["correct"] += int(is_correct)
                        post_merge_summary[tool_name]["total"] += 1

            # Compute accuracy per tool
            for tool_name, agg in pre_merge_summary.items():
                total = agg.get("total", 0)
                correct = agg.get("correct", 0)
                agg["accuracy"] = (correct / total) if total else 0.0

            for tool_name, agg in post_merge_summary.items():
                total = agg.get("total", 0)
                correct = agg.get("correct", 0)
                agg["accuracy"] = (correct / total) if total else 0.0

            # Final summary
            successful_count = len([m for m in all_metadata if "error" not in m])
            error_count = len([m for m in all_metadata if "error" in m])
            logger.info("=== EXPERIMENT COMPLETE ===")
            logger.info(f"Total questions processed: {len(results)}")
            logger.info(f"Successful runs: {successful_count}")
            logger.info(f"Errors: {error_count}")
            logger.info(f"Expected sample size: {dataset_size}")
            logger.info("")
            logger.info("PRE-MERGE RESULTS SUMMARY:")
            for tool_name, stats in pre_merge_summary.items():
                logger.info(f"  {tool_name}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.3f})")
            logger.info("")
            logger.info("POST-MERGE RESULTS SUMMARY:")
            for tool_name, stats in post_merge_summary.items():
                logger.info(f"  {tool_name}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.3f})")

            return {
                "experiment_id": SlidersGlobal.experiment_id,
                "results": results,
                "all_metadata": all_metadata,
                "pre_merge_summary": pre_merge_summary,
                "post_merge_summary": post_merge_summary,
            }
        else:
            # Original single evaluation path
            results_summary = {}
            for result in results:
                for tool_name, tool_data in result.get("evaluation_tools", {}).items():
                    if tool_name not in results_summary:
                        results_summary[tool_name] = {"correct": 0, "total": 0}

                    is_correct = False
                    if isinstance(tool_data, dict):
                        is_correct = bool(tool_data.get("correct", False))

                    results_summary[tool_name]["correct"] += int(is_correct)
                    results_summary[tool_name]["total"] += 1

            # compute accuracy per tool
            for tool_name, agg in results_summary.items():
                total = agg.get("total", 0)
                correct = agg.get("correct", 0)
                agg["accuracy"] = (correct / total) if total else 0.0

            # Final summary
            successful_count = len([m for m in all_metadata if "error" not in m])
            error_count = len([m for m in all_metadata if "error" in m])
            logger.info("=== EXPERIMENT COMPLETE ===")
            logger.info(f"Total questions processed: {len(results)}")
            logger.info(f"Successful runs: {successful_count}")
            logger.info(f"Errors: {error_count}")
            logger.info(f"Expected sample size: {dataset_size}")

            return {
                "experiment_id": SlidersGlobal.experiment_id,
                "results": results,
                "all_metadata": all_metadata,
                "results_summary": results_summary,
            }
