#!/usr/bin/env python3
"""Batch-convert FinanceBench PDFs to Markdown using Docling.

Reuses ``sliders.run._convert_pdf_to_markdown`` so the output matches what the
main pipeline produces. The loop is resilient: each PDF is converted in its own
try/except, so one bad/unparseable document is logged and skipped instead of
aborting the rest. PDFs that already have a non-empty ``.md`` are skipped, which
makes the script safe to re-run / resume.

Usage:
    uv run python convert_financebench_pdfs.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

from sliders.run import _convert_pdf_to_markdown

PDF_DIR = Path("datasets/financebench/pdfs")
MD_DIR = Path("datasets/financebench/markdown")


def main() -> int:
    MD_DIR.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {PDF_DIR.resolve()}", flush=True)
        return 1

    todo = []
    for pdf in pdfs:
        md_path = MD_DIR / (pdf.stem + ".md")
        if md_path.exists() and md_path.stat().st_size > 0:
            continue
        todo.append(pdf)

    print(
        f"{len(pdfs)} PDFs found; {len(pdfs) - len(todo)} already converted; "
        f"{len(todo)} to convert.",
        flush=True,
    )

    converted: list[str] = []
    failed: list[str] = []
    for i, pdf in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] Converting {pdf.name} ...", flush=True)
        try:
            out_path = _convert_pdf_to_markdown(pdf, MD_DIR)
            size = out_path.stat().st_size
            print(f"    -> {out_path.name} ({size:,} bytes)", flush=True)
            converted.append(pdf.stem)
        except Exception as e:  # noqa: BLE001 - one bad PDF must not abort the rest
            print(f"    !! FAILED {pdf.name}: {e}", flush=True)
            traceback.print_exc()
            failed.append(pdf.stem)

    print("\n=== Conversion summary ===", flush=True)
    print(f"Converted: {len(converted)}", flush=True)
    print(f"Failed:    {len(failed)}", flush=True)
    if failed:
        print("Failed docs: " + ", ".join(sorted(failed)), flush=True)

    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
