"""Generate a self-contained HTML report that visualizes every SLIDERS step.

The report walks through the pipeline in order -- contextualized chunking,
schema induction, contextualized extraction (with provenance), data
reconciliation, and SQL answer synthesis -- so that each step's inputs and
outputs are visible for a single run.

Only produced when ``visualize`` is enabled. Purely additive: it reads
already-computed, in-memory objects plus the run ``metadata`` and never
mutates pipeline state. Rendering is hand-written HTML with inline CSS/JS
(no extra dependencies).
"""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from sliders.log_utils import logger

# Caps to keep the report readable for very large runs.
MAX_TABLE_ROWS = 200
MAX_EXTRACTION_ROWS = 150
MAX_CHUNK_CHARS = 20000


# ---------------------------------------------------------------------------
# Serialization helpers (turn in-memory objects into a JSON-safe trace)
# ---------------------------------------------------------------------------
def _to_native(value: Any) -> Any:
    """Coerce numpy scalars / NaN into JSON-serializable Python values."""
    if value is None:
        return None
    # numpy scalars expose .item()
    if hasattr(value, "item") and not isinstance(value, (str, bytes, list, tuple, dict)):
        try:
            value = value.item()
        except Exception:
            pass
    # NaN != NaN
    if isinstance(value, float) and value != value:
        return None
    return value


def _df_to_serializable(df: Any, max_rows: int = MAX_TABLE_ROWS) -> dict | None:
    """Convert a pandas DataFrame into ``{columns, rows, total_rows, shown_rows}``."""
    if df is None:
        return None
    try:
        if getattr(df, "empty", False):
            return {"columns": [str(c) for c in df.columns], "rows": [], "total_rows": 0, "shown_rows": 0}
        columns = [str(c) for c in df.columns]
        total = int(len(df))
        shown_df = df.head(max_rows)
        rows: list[list[Any]] = []
        for _, record in shown_df.iterrows():
            rows.append([_to_native(record[c]) for c in df.columns])
        return {"columns": columns, "rows": rows, "total_rows": total, "shown_rows": len(rows)}
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug(f"[visualize] Could not serialize dataframe: {exc}")
        return None


def _schema_to_dict(schema: Any) -> dict | None:
    """Return a plain dict for a ``Tables`` schema object."""
    if schema is None:
        return None
    if hasattr(schema, "model_dump"):
        try:
            return schema.model_dump()
        except Exception:
            pass
    if isinstance(schema, dict):
        return schema
    return None


def _extracted_tables_to_serializable(extracted_tables: Any) -> list[dict]:
    """Normalize the raw extraction dict into per-table, per-row records.

    Input shape: ``{table_name: [{"__metadata__": {...}, "fields": {name: {value, quote, ...}}}]}``
    """
    result: list[dict] = []
    if not extracted_tables or not isinstance(extracted_tables, dict):
        return result
    for table_name, raw_rows in extracted_tables.items():
        rows: list[dict] = []
        for raw in raw_rows or []:
            if not isinstance(raw, dict):
                continue
            meta = raw.get("__metadata__", {}) or {}
            fields = {}
            for field_name, payload in (raw.get("fields", {}) or {}).items():
                if isinstance(payload, dict):
                    fields[field_name] = {
                        "value": _to_native(payload.get("value")),
                        "quote": payload.get("quote"),
                        "rationale": payload.get("rationale"),
                        "confidence": payload.get("confidence"),
                        "is_explicit": payload.get("is_explicit"),
                    }
                else:
                    fields[field_name] = {"value": _to_native(payload)}
            rows.append(
                {
                    "document_name": meta.get("document_name"),
                    "chunk_id": meta.get("chunk_id"),
                    "is_placeholder": bool(meta.get("is_placeholder", False)),
                    "fields": fields,
                }
            )
        result.append({"name": table_name, "row_count": len(rows), "rows": rows})
    return result


def _tables_to_map(tables: Any) -> dict[str, dict]:
    """Map ExtractedTable.name -> serialized dataframe."""
    out: dict[str, dict] = {}
    for table in tables or []:
        try:
            name = getattr(table, "name", None)
            if name is None:
                continue
            out[name] = _df_to_serializable(getattr(table, "dataframe", None))
        except Exception:
            continue
    return out


def _load_reconciliation_stats(out_dir: Path, metadata: dict) -> dict:
    """Best-effort load of the accumulated reconciliation_stats.json (keyed by table)."""
    filename = "reconciliation_stats.json"
    try:
        filename = (
            metadata.get("config", {})
            .get("system_config", {})
            .get("merge_tables", {})
            .get("reconciliation", {})
            .get("statistics", {})
            .get("filename", filename)
        )
    except Exception:
        pass
    stats_path = Path(out_dir) / filename
    if stats_path.exists():
        try:
            with open(stats_path, "r") as f:
                return json.load(f)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(f"[visualize] Could not read reconciliation stats: {exc}")
    return {}


def _build_trace(
    *,
    out_dir: Path,
    question: str,
    question_id: str,
    documents: list,
    schema: Any,
    extracted_tables: Any,
    pre_merge_tables: Any,
    post_merge_tables: Any,
    metadata: dict,
    final_answer: Any,
) -> dict:
    """Assemble a JSON-serializable trace of the full pipeline for one question."""
    metadata = metadata or {}
    answer_meta = metadata.get("answer_generation", {}) or {}
    extraction_meta = metadata.get("extraction", {}) or {}
    irrelevant_chunks = extraction_meta.get("irrelevant_chunks", {}) or {}

    # --- Chunking ---
    chunk_docs = []
    for doc in documents or []:
        try:
            doc_name = getattr(doc, "document_name", "document")
            doc_irrelevant = set(irrelevant_chunks.get(doc_name, []) or [])
            chunks = []
            for idx, chunk in enumerate(getattr(doc, "chunks", []) or []):
                content = chunk.get("content", "") if isinstance(chunk, dict) else str(chunk)
                headers = chunk.get("metadata", {}) if isinstance(chunk, dict) else {}
                truncated = len(content) > MAX_CHUNK_CHARS
                relevant = None
                if doc_name in irrelevant_chunks:
                    relevant = idx not in doc_irrelevant
                chunks.append(
                    {
                        "index": idx,
                        "headers": headers,
                        "length": len(content),
                        "content": content[:MAX_CHUNK_CHARS],
                        "truncated": truncated,
                        "relevant": relevant,
                    }
                )
            chunk_docs.append(
                {
                    "name": doc_name,
                    "description": getattr(doc, "description", None),
                    "size": len(getattr(doc, "content", "") or ""),
                    "num_chunks": len(chunks),
                    "chunks": chunks,
                }
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(f"[visualize] Failed to serialize a document: {exc}")

    # --- Reconciliation: pair pre/post dataframes + per-table stats ---
    recon_stats = _load_reconciliation_stats(out_dir, metadata)
    before_map = _tables_to_map(pre_merge_tables)
    after_map = _tables_to_map(post_merge_tables)
    recon_meta = metadata.get("reconciliation", {}) or {}
    table_names = list(dict.fromkeys(list(before_map.keys()) + list(after_map.keys())))
    recon_tables = []
    for name in table_names:
        recon_tables.append(
            {
                "name": name,
                "stats": recon_stats.get(name),
                "before": before_map.get(name),
                "after": after_map.get(name),
            }
        )

    trace = {
        "question": question,
        "question_id": question_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "final_answer": final_answer if final_answer is not None else metadata.get("post_merge_answer"),
        "overview": {
            "num_documents": metadata.get("num_documents", len(documents or [])),
            "document_names": metadata.get("document_names", []),
            "document_sizes": metadata.get("document_sizes", []),
            "total_chunks": metadata.get("total_chunks"),
            "timing": metadata.get("timing", {}),
            "errors": metadata.get("errors", []),
        },
        "chunking": {"documents": chunk_docs},
        "schema": {
            "schema": metadata.get("schema", {}).get("schema_object") or _schema_to_dict(schema),
            "question_type": metadata.get("schema", {}).get("question_type"),
            "document_type": metadata.get("schema", {}).get("document_type"),
            "classification_reasoning": metadata.get("schema", {}).get("classification_reasoning"),
            "rephrase": metadata.get("rephrase_question"),
            "generated_classes": metadata.get("schema", {}).get("generated_classes"),
            "total_fields": metadata.get("schema", {}).get("total_fields"),
        },
        "extraction": {
            "stats": {
                k: extraction_meta.get(k)
                for k in (
                    "chunks_processed",
                    "successful_extractions",
                    "failed_extractions",
                    "retry_attempts",
                    "success_rate",
                    "extraction_time",
                )
            },
            "tables": _extracted_tables_to_serializable(extracted_tables),
        },
        "reconciliation": {
            "primary_key": recon_meta.get("primary_key"),
            "pk_groups": recon_meta.get("pk_groups"),
            "all_operations": recon_meta.get("all_operations"),
            "tables": recon_tables,
        },
        "answer": {
            "final_answer": final_answer if final_answer is not None else metadata.get("post_merge_answer"),
            "query_history": answer_meta.get("query_history"),
            "regular_answer": metadata.get("regular_answer"),
            "inspect_answer": metadata.get("inspect_answer"),
            "pre_merge_answer": metadata.get("pre_merge_answer"),
            "post_merge_answer": metadata.get("post_merge_answer"),
            "citation_sql": answer_meta.get("citation_sql"),
            "citation_paragraph": answer_meta.get("citation_paragraph"),
            "reconciliation_stats_summary": answer_meta.get("reconciliation_stats_summary"),
            "num_queries": answer_meta.get("num_queries"),
            "used_tables": answer_meta.get("used_tables"),
        },
    }
    return trace


# ---------------------------------------------------------------------------
# HTML rendering helpers
# ---------------------------------------------------------------------------
def esc(value: Any) -> str:
    """HTML-escape any value as a string."""
    if value is None:
        return ""
    return html.escape(str(value))


def _fmt_value(value: Any) -> str:
    """Render a table-cell value, handling None/empty/lists/dicts."""
    if value is None or (isinstance(value, str) and value == ""):
        return '<span class="muted">&mdash;</span>'
    if isinstance(value, bool):
        return esc("true" if value else "false")
    if isinstance(value, (list, tuple)):
        return esc(", ".join(str(x) for x in value))
    if isinstance(value, dict):
        return esc(json.dumps(value, default=str))
    return esc(value)


def _badge(text: str, kind: str = "") -> str:
    cls = f"badge {kind}".strip()
    return f'<span class="{cls}">{esc(text)}</span>'


def _details(summary_html: str, body_html: str, open_: bool = False) -> str:
    open_attr = " open" if open_ else ""
    return f'<details{open_attr}><summary>{summary_html}</summary><div class="details-body">{body_html}</div></details>'


def _code_block(text: Any, lang: str = "sql") -> str:
    if text is None or str(text).strip() == "":
        return '<p class="muted">&mdash;</p>'
    return f'<pre class="code {esc(lang)}"><code>{esc(text)}</code></pre>'


def _answer_block(text: Any) -> str:
    if text is None or str(text).strip() == "":
        return '<p class="muted">No answer produced.</p>'
    return f'<div class="answer-body">{esc(text)}</div>'


def _render_data_table(data: dict | None, empty_msg: str = "No data.", max_cols: int | None = None) -> str:
    """Render a ``{columns, rows, total_rows, shown_rows}`` dict as an HTML table."""
    if not data or not data.get("columns"):
        return f'<p class="muted">{esc(empty_msg)}</p>'
    columns = data["columns"]
    rows = data.get("rows", [])
    col_note = ""
    if max_cols is not None and len(columns) > max_cols:
        col_note = f'<p class="muted">Showing first {max_cols} of {len(columns)} columns.</p>'
        keep = max_cols
        columns = columns[:keep]
        rows = [r[:keep] for r in rows]
    thead = "".join(f"<th>{esc(c)}</th>" for c in columns)
    body_rows = []
    for record in rows:
        cells = "".join(f"<td>{_fmt_value(v)}</td>" for v in record)
        body_rows.append(f"<tr>{cells}</tr>")
    tbody = "".join(body_rows) or f'<tr><td colspan="{len(columns)}" class="muted">No rows.</td></tr>'
    note = ""
    total = data.get("total_rows")
    shown = data.get("shown_rows")
    if total is not None and shown is not None and total > shown:
        note = f'<p class="muted">Showing first {shown} of {total} rows.</p>'
    return (
        f'<div class="table-wrap"><table><thead><tr>{thead}</tr></thead>'
        f"<tbody>{tbody}</tbody></table></div>{note}{col_note}"
    )


def _render_kv(pairs: list[tuple[str, Any]]) -> str:
    """Render a definition-list-style key/value block."""
    items = []
    for key, value in pairs:
        if value is None or value == "":
            continue
        items.append(
            f'<div class="kv"><span class="kv-key">{esc(key)}</span>'
            f'<span class="kv-val">{_fmt_value(value)}</span></div>'
        )
    if not items:
        return ""
    return f'<div class="kv-grid">{"".join(items)}</div>'


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------
def _render_overview(trace: dict) -> str:
    ov = trace.get("overview", {})
    timing = ov.get("timing", {}) or {}
    doc_names = ov.get("document_names") or []
    doc_sizes = ov.get("document_sizes") or []
    doc_rows = []
    for i, name in enumerate(doc_names):
        size = doc_sizes[i] if i < len(doc_sizes) else None
        doc_rows.append((name, f"{size:,} chars" if isinstance(size, int) else "&mdash;"))

    def _t(section: str, key: str = "total_duration"):
        if section == "total":
            val = timing.get("total_duration")
        else:
            val = (timing.get(section, {}) or {}).get(key)
        return f"{val:.2f}s" if isinstance(val, (int, float)) else None

    stat_cards = [
        ("Documents", ov.get("num_documents")),
        ("Total chunks", ov.get("total_chunks")),
        ("Total time", _t("total")),
        ("Schema time", _t("schema_generation", "generation_time")),
        ("Merge time", _t("table_merging", "merging_time")),
    ]
    cards_html = "".join(
        f'<div class="stat"><div class="stat-val">{_fmt_value(v)}</div><div class="stat-label">{esc(k)}</div></div>'
        for k, v in stat_cards
        if v is not None
    )
    docs_html = _render_kv(doc_rows) if doc_rows else ""
    errors = ov.get("errors") or []
    errors_html = ""
    if errors:
        err_items = "".join(f"<li>{esc(e.get('stage', '?'))}: {esc(e.get('error', ''))}</li>" for e in errors)
        errors_html = (
            f'<div class="callout warn"><strong>{len(errors)} error(s) recorded</strong><ul>{err_items}</ul></div>'
        )

    return (
        f'<section id="overview" class="step step-overview">'
        f"<h2>Run overview</h2>"
        f'<div class="question-box"><span class="q-label">Question</span>'
        f'<div class="q-text">{esc(trace.get("question"))}</div></div>'
        f'<div class="stat-row">{cards_html}</div>'
        f"{docs_html}"
        f"{errors_html}"
        f'<div class="callout final-answer"><span class="q-label">Final answer</span>'
        f"{_answer_block(trace.get('final_answer'))}</div>"
        f"</section>"
    )


def _render_chunking(trace: dict) -> str:
    docs = (trace.get("chunking", {}) or {}).get("documents", [])
    if not docs:
        return (
            '<section id="chunking" class="step step-chunking"><h2>1 &middot; Contextualized chunking</h2>'
            '<p class="muted">No chunk data captured.</p></section>'
        )
    doc_blocks = []
    for doc in docs:
        chunk_cards = []
        for ch in doc.get("chunks", []):
            relevant = ch.get("relevant")
            if relevant is True:
                gate = _badge("relevant", "ok")
            elif relevant is False:
                gate = _badge("gated out", "muted-badge")
            else:
                gate = ""
            headers = ch.get("headers", {}) or {}
            header_txt = headers.get("headers") if isinstance(headers, dict) else None
            if not header_txt and isinstance(headers, dict):
                header_txt = " > ".join(str(v) for v in headers.values() if v)
            content = ch.get("content", "")
            if ch.get("truncated"):
                content = content + "\n\n... (truncated)"
            summary = (
                f'<span class="chunk-idx">Chunk {ch.get("index")}</span> '
                f'{gate} <span class="muted">{ch.get("length", 0):,} chars</span>'
                + (f' <span class="chunk-head">{esc(header_txt)}</span>' if header_txt else "")
            )
            chunk_cards.append(_details(summary, f'<pre class="chunk-content">{esc(content)}</pre>'))
        desc = doc.get("description")
        desc_html = (
            f'<div class="doc-desc"><span class="q-label">View / description</span> {esc(desc)}</div>' if desc else ""
        )
        doc_blocks.append(
            f'<div class="doc-block"><h3>{esc(doc.get("name"))} '
            f'<span class="muted">({doc.get("num_chunks", 0)} chunks, {doc.get("size", 0):,} chars)</span></h3>'
            f"{desc_html}"
            f'<div class="chunk-list">{"".join(chunk_cards)}</div></div>'
        )
    return (
        '<section id="chunking" class="step step-chunking">'
        "<h2>1 &middot; Contextualized chunking</h2>"
        '<p class="step-desc">Each document is augmented with a description ("view") and split into locally '
        "self-contained chunks. Chunks marked <em>gated out</em> were judged not relevant to the question and "
        "skipped during extraction.</p>"
        f"{''.join(doc_blocks)}</section>"
    )


def _render_schema(trace: dict) -> str:
    s = trace.get("schema", {}) or {}
    schema_obj = s.get("schema") or {}
    tables = schema_obj.get("tables", []) if isinstance(schema_obj, dict) else []
    table_blocks = []
    for table in tables:
        fields = table.get("fields", []) or []
        rows = []
        for f in fields:
            enum_vals = f.get("enum_values")
            rows.append(
                [
                    f.get("name"),
                    f.get("data_type"),
                    ", ".join(enum_vals) if enum_vals else None,
                    f.get("unit"),
                    "yes" if f.get("required") else "no",
                    f.get("description"),
                ]
            )
        field_table = _render_data_table(
            {
                "columns": ["Field", "Type", "Enum values", "Unit", "Required", "Description"],
                "rows": rows,
                "total_rows": len(rows),
                "shown_rows": len(rows),
            },
            empty_msg="No fields.",
        )
        table_blocks.append(
            f'<div class="schema-table"><h3>{esc(table.get("name"))}</h3>'
            f'<p class="muted">{esc(table.get("description"))}</p>{field_table}</div>'
        )

    classification = _render_kv(
        [
            ("Question type", s.get("question_type")),
            ("Document type", s.get("document_type")),
            ("Tables", s.get("generated_classes")),
            ("Total fields", s.get("total_fields")),
        ]
    )
    class_reason = s.get("classification_reasoning")
    class_reason_html = _details("Classification reasoning", f"<p>{esc(class_reason)}</p>") if class_reason else ""
    schema_reason = schema_obj.get("reasoning") if isinstance(schema_obj, dict) else None
    schema_reason_html = _details("Schema reasoning", f"<p>{esc(schema_reason)}</p>") if schema_reason else ""

    rephrase = s.get("rephrase") or {}
    rephrase_html = ""
    if isinstance(rephrase, dict) and rephrase.get("enabled") and rephrase.get("questions"):
        rq = rephrase["questions"]
        if isinstance(rq, dict):
            rephrase_html = _details(
                "Per-stage rephrased questions",
                _render_kv([(k.replace("_", " ").title(), v) for k, v in rq.items()]),
            )

    if not table_blocks:
        table_blocks = ['<p class="muted">No schema captured.</p>']
    return (
        '<section id="schema" class="step step-schema">'
        "<h2>2 &middot; Schema induction</h2>"
        '<p class="step-desc">A question- and document-aware relational schema is induced. Extraction later fills '
        "these tables, one row per finding.</p>"
        f"{classification}{class_reason_html}{schema_reason_html}{rephrase_html}"
        f"{''.join(table_blocks)}</section>"
    )


def _quote_text(quote: Any) -> str:
    if quote is None:
        return ""
    if isinstance(quote, (list, tuple)):
        return " ... ".join(str(q) for q in quote if q)
    return str(quote)


def _render_extraction(trace: dict) -> str:
    ex = trace.get("extraction", {}) or {}
    stats = ex.get("stats", {}) or {}
    tables = ex.get("tables", []) or []
    stats_html = _render_kv(
        [
            ("Chunks processed", stats.get("chunks_processed")),
            ("Successful extractions", stats.get("successful_extractions")),
            ("Failed extractions", stats.get("failed_extractions")),
            ("Retries", stats.get("retry_attempts")),
            (
                "Extraction time",
                f"{stats.get('extraction_time'):.2f}s"
                if isinstance(stats.get("extraction_time"), (int, float))
                else None,
            ),
        ]
    )

    table_blocks = []
    for table in tables:
        rows = table.get("rows", [])
        # Determine field ordering across rows.
        field_names: list[str] = []
        for r in rows:
            for fname in r.get("fields", {}).keys():
                if fname not in field_names:
                    field_names.append(fname)

        # Compact overview table: source + each field value (with provenance tooltip).
        header_cells = "".join(f"<th>{esc(fn)}</th>" for fn in field_names)
        body_rows = []
        provenance_cards = []
        for i, r in enumerate(rows[:MAX_EXTRACTION_ROWS]):
            src = f"{r.get('document_name') or '?'} &middot; chunk {r.get('chunk_id')}"
            if r.get("is_placeholder"):
                src += " " + _badge("placeholder", "muted-badge")
            value_cells = []
            prov_rows = []
            for fn in field_names:
                fdata = r.get("fields", {}).get(fn, {}) or {}
                value = fdata.get("value")
                conf = fdata.get("confidence")
                rationale = fdata.get("rationale")
                quote = _quote_text(fdata.get("quote"))
                tip_parts = []
                if conf:
                    tip_parts.append(f"Confidence: {conf}")
                if rationale:
                    tip_parts.append(f"Rationale: {rationale}")
                if quote:
                    tip_parts.append(f"Quote: {quote}")
                tip = esc("\n".join(tip_parts))
                value_cells.append(f'<td title="{tip}">{_fmt_value(value)}</td>')
                prov_rows.append([fn, value, conf, quote or None, rationale])
            body_rows.append(f'<tr><td class="src">{src}</td>{"".join(value_cells)}</tr>')
            prov_table = _render_data_table(
                {
                    "columns": ["Field", "Value", "Confidence", "Provenance quote", "Rationale"],
                    "rows": prov_rows,
                    "total_rows": len(prov_rows),
                    "shown_rows": len(prov_rows),
                },
                empty_msg="No fields.",
            )
            provenance_cards.append(_details(f'Row {i + 1} &mdash; <span class="src">{src}</span>', prov_table))

        note = ""
        if len(rows) > MAX_EXTRACTION_ROWS:
            note = f'<p class="muted">Showing first {MAX_EXTRACTION_ROWS} of {len(rows)} rows.</p>'
        if rows:
            overview_table = (
                f'<div class="table-wrap"><table><thead><tr><th>Source</th>{header_cells}</tr></thead>'
                f"<tbody>{''.join(body_rows)}</tbody></table></div>{note}"
            )
            provenance_html = _details("Show full provenance (quotes &amp; rationale)", "".join(provenance_cards))
        else:
            overview_table = '<p class="muted">No rows extracted for this table.</p>'
            provenance_html = ""

        table_blocks.append(
            f'<div class="extract-table"><h3>{esc(table.get("name"))} '
            f'<span class="muted">({table.get("row_count", 0)} rows)</span></h3>'
            f'<p class="muted">Hover a value to see its confidence, provenance quote, and rationale.</p>'
            f"{overview_table}{provenance_html}</div>"
        )

    if not table_blocks:
        table_blocks = ['<p class="muted">No extraction data captured.</p>']
    return (
        '<section id="extraction" class="step step-extraction">'
        "<h2>3 &middot; Contextualized extraction &rarr; relational DB</h2>"
        '<p class="step-desc">Each relevant chunk is read and its salient information is written into the schema '
        "tables. Every cell keeps its source chunk, a provenance quote, a rationale, and a confidence. These are the "
        "<em>pre-reconciliation</em> rows (one row per chunk-level extraction).</p>"
        f"{stats_html}{''.join(table_blocks)}</section>"
    )


def _render_reconciliation(trace: dict) -> str:
    recon = trace.get("reconciliation", {}) or {}
    tables = recon.get("tables", []) or []
    if not tables:
        return (
            '<section id="reconciliation" class="step step-reconciliation">'
            "<h2>4 &middot; Data reconciliation</h2>"
            '<p class="muted">No reconciliation data captured (merging may be disabled).</p></section>'
        )
    table_blocks = []
    for table in tables:
        stats = table.get("stats") or {}
        pk = stats.get("primary_key") or (recon.get("primary_key", {}) or {}).get("fields")
        row_grouping = stats.get("row_grouping", {}) or {}
        controller_actions = stats.get("controller_actions", {}) or {}
        final_stats = stats.get("final_stats", {}) or {}
        summary = _render_kv(
            [
                ("Primary key", ", ".join(pk) if isinstance(pk, list) else pk),
                ("Unique PK values", row_grouping.get("unique_primary_key_values")),
                ("Single-row groups", row_grouping.get("single_row_groups")),
                ("Multi-row groups", row_grouping.get("multi_row_groups")),
                ("Rows before", final_stats.get("initial_total_rows")),
                ("Rows after", final_stats.get("final_total_rows")),
                ("Rows reduced", final_stats.get("total_rows_reduced")),
            ]
        )
        ops = {k: v for k, v in controller_actions.items() if v}
        ops_html = ""
        if ops:
            ops_badges = " ".join(_badge(f"{k}: {v}", "ok") for k, v in ops.items())
            ops_html = f'<div class="ops-row"><span class="q-label">Operations</span> {ops_badges}</div>'

        before = table.get("before")
        after = table.get("after")
        before_html = _render_data_table(before, empty_msg="No pre-reconciliation table.", max_cols=12)
        after_html = _render_data_table(after, empty_msg="No post-reconciliation table.", max_cols=12)
        compare = (
            '<div class="compare-grid">'
            f'<div class="compare-col"><h4>Before reconciliation</h4>{before_html}</div>'
            f'<div class="compare-col"><h4>After reconciliation</h4>{after_html}</div>'
            "</div>"
        )
        table_blocks.append(
            f'<div class="recon-table"><h3>{esc(table.get("name"))}</h3>{summary}{ops_html}{compare}</div>'
        )
    return (
        '<section id="reconciliation" class="step step-reconciliation">'
        "<h2>4 &middot; Data reconciliation</h2>"
        '<p class="step-desc">An SQL agent picks a primary key, groups rows by it, and deduplicates, aggregates, and '
        "resolves conflicts. Each surviving row records a reconciliation context (see the "
        "<code>__reconciliation_context__</code> column) describing what was merged or discarded.</p>"
        f"{''.join(table_blocks)}</section>"
    )


def _render_answer(trace: dict) -> str:
    ans = trace.get("answer", {}) or {}
    query_history = ans.get("query_history") or []
    query_blocks = []
    for i, step in enumerate(query_history):
        if not isinstance(step, dict):
            continue
        reasoning = step.get("reasoning")
        sql = step.get("sql")
        result = step.get("result")
        body = (
            (f'<div class="reasoning">{esc(reasoning)}</div>' if reasoning else "")
            + _code_block(sql, "sql")
            + (f'<div class="sql-result">{_code_block(result, "text")}</div>' if result else "")
        )
        query_blocks.append(_details(f"Query {i + 1}", body, open_=(i == 0)))
    queries_html = "".join(query_blocks) if query_blocks else '<p class="muted">No SQL query history captured.</p>'

    citation_sql = ans.get("citation_sql")
    citation_paragraph = ans.get("citation_paragraph")
    citation_html = ""
    if citation_sql or citation_paragraph:
        citation_html = _details(
            "Citations / provenance",
            (_code_block(citation_sql, "sql") if citation_sql else "")
            + (f'<div class="answer-body">{esc(citation_paragraph)}</div>' if citation_paragraph else ""),
        )

    recon_summary = ans.get("reconciliation_stats_summary")
    recon_summary_html = (
        _details("Reconciliation summary", f'<div class="answer-body">{esc(recon_summary)}</div>')
        if recon_summary
        else ""
    )

    # Alternative answers for comparison.
    alt_blocks = []
    for label, key in (
        ("Pre-merge answer", "pre_merge_answer"),
        ("Regular answer (no inspect)", "regular_answer"),
    ):
        val = ans.get(key)
        if val:
            alt_blocks.append(_details(label, _answer_block(val)))
    alt_html = _details("Alternative answers (for comparison)", "".join(alt_blocks)) if alt_blocks else ""

    return (
        '<section id="answer" class="step step-answer">'
        "<h2>5 &middot; SQL answer synthesis (query)</h2>"
        '<p class="step-desc">An answer agent iteratively writes SQL against the reconciled database, inspects the '
        "results, and composes the final natural-language answer.</p>"
        f'<div class="q-label">SQL queries run against the reconciled tables</div>{queries_html}'
        f'<div class="callout final-answer"><span class="q-label">Final answer</span>'
        f"{_answer_block(ans.get('final_answer'))}</div>"
        f"{citation_html}{recon_summary_html}{alt_html}</section>"
    )


# ---------------------------------------------------------------------------
# Page assembly
# ---------------------------------------------------------------------------
_CSS = """
:root {
  --bg: #0f1117; --panel: #171a23; --panel-2: #1e222d; --border: #2a2f3d;
  --text: #e6e8ee; --muted: #9aa3b2; --accent: #6ea8fe;
  --overview: #6ea8fe; --chunking: #4dabf7; --schema: #b197fc;
  --extraction: #63e6be; --reconciliation: #ffa94d; --answer: #ff8787;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0; background: var(--bg); color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  line-height: 1.5;
}
a { color: var(--accent); text-decoration: none; }
header.top {
  padding: 28px 32px 20px; border-bottom: 1px solid var(--border);
  background: linear-gradient(180deg, #131722, #0f1117);
}
header.top h1 { margin: 0 0 4px; font-size: 22px; letter-spacing: 0.5px; }
header.top .sub { color: var(--muted); font-size: 13px; }
nav.steps {
  position: sticky; top: 0; z-index: 10; display: flex; flex-wrap: wrap; gap: 8px;
  padding: 12px 32px; background: rgba(15,17,23,0.92); backdrop-filter: blur(6px);
  border-bottom: 1px solid var(--border);
}
nav.steps a {
  font-size: 13px; padding: 6px 12px; border-radius: 999px; color: var(--text);
  background: var(--panel-2); border: 1px solid var(--border);
}
nav.steps a:hover { border-color: var(--accent); }
nav.steps .spacer { flex: 1; }
nav.steps button {
  font-size: 12px; padding: 6px 12px; border-radius: 8px; cursor: pointer;
  color: var(--text); background: var(--panel-2); border: 1px solid var(--border);
}
main { padding: 8px 32px 80px; max-width: 1200px; margin: 0 auto; }
.step { padding: 24px 0 8px; border-top: 1px solid var(--border); scroll-margin-top: 64px; }
.step:first-of-type { border-top: none; }
.step > h2 {
  font-size: 19px; margin: 0 0 6px; padding-left: 12px; border-left: 4px solid var(--accent);
}
.step-overview > h2 { border-color: var(--overview); }
.step-chunking > h2 { border-color: var(--chunking); }
.step-schema > h2 { border-color: var(--schema); }
.step-extraction > h2 { border-color: var(--extraction); }
.step-reconciliation > h2 { border-color: var(--reconciliation); }
.step-answer > h2 { border-color: var(--answer); }
.step-desc { color: var(--muted); font-size: 14px; margin: 0 0 16px; max-width: 900px; }
h3 { font-size: 16px; margin: 20px 0 8px; }
h4 { font-size: 13px; margin: 0 0 8px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; }
.muted { color: var(--muted); }
.q-label { display: inline-block; font-size: 11px; text-transform: uppercase; letter-spacing: 0.6px; color: var(--muted); margin-bottom: 4px; }
.question-box { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 16px 18px; margin: 12px 0; }
.q-text { font-size: 18px; font-weight: 600; }
.stat-row { display: flex; flex-wrap: wrap; gap: 12px; margin: 16px 0; }
.stat { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 14px 18px; min-width: 120px; }
.stat-val { font-size: 22px; font-weight: 700; }
.stat-label { font-size: 12px; color: var(--muted); margin-top: 2px; }
.callout { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 16px 18px; margin: 16px 0; }
.callout.final-answer { border-left: 4px solid var(--answer); }
.callout.warn { border-left: 4px solid var(--reconciliation); }
.answer-body { white-space: pre-wrap; font-size: 14px; }
.kv-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 8px 16px; margin: 12px 0; }
.kv { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 8px 12px; }
.kv-key { display: block; font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.4px; }
.kv-val { font-size: 14px; word-break: break-word; }
.doc-block, .schema-table, .extract-table, .recon-table { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 16px 18px; margin: 16px 0; }
.doc-desc { color: var(--text); font-size: 14px; margin-bottom: 12px; }
.chunk-list { display: flex; flex-direction: column; gap: 6px; }
.chunk-idx { font-weight: 600; }
.chunk-head { color: var(--muted); font-size: 12px; }
.chunk-content, .code { background: var(--panel-2); border: 1px solid var(--border); border-radius: 8px; padding: 12px; overflow-x: auto; font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace; font-size: 12.5px; white-space: pre-wrap; word-break: break-word; margin: 8px 0; }
pre.code code { white-space: pre-wrap; }
.badge { display: inline-block; font-size: 11px; padding: 2px 8px; border-radius: 999px; background: var(--panel-2); border: 1px solid var(--border); }
.badge.ok { background: rgba(99,230,190,0.15); border-color: rgba(99,230,190,0.4); color: #63e6be; }
.badge.muted-badge { color: var(--muted); }
details { background: var(--panel-2); border: 1px solid var(--border); border-radius: 8px; padding: 0; margin: 6px 0; }
details > summary { cursor: pointer; padding: 10px 12px; font-size: 13px; user-select: none; list-style: none; }
details > summary::-webkit-details-marker { display: none; }
details > summary::before { content: "\\25B8"; display: inline-block; margin-right: 8px; color: var(--muted); transition: transform 0.15s; }
details[open] > summary::before { transform: rotate(90deg); }
.details-body { padding: 0 12px 12px; }
.table-wrap { overflow-x: auto; margin: 8px 0; border: 1px solid var(--border); border-radius: 8px; }
table { border-collapse: collapse; width: 100%; font-size: 12.5px; }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); vertical-align: top; }
th { background: var(--panel-2); position: sticky; top: 0; font-weight: 600; }
tbody tr:nth-child(even) { background: rgba(255,255,255,0.02); }
td.src, .src { color: var(--muted); white-space: nowrap; }
.compare-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
@media (max-width: 900px) { .compare-grid { grid-template-columns: 1fr; } }
.reasoning { color: var(--muted); font-size: 13px; margin: 6px 0; }
.ops-row { margin: 8px 0; }
.callout ul { margin: 8px 0 0; padding-left: 18px; }
"""

_JS = """
function toggleAll(open) {
  document.querySelectorAll('details').forEach(function (d) { d.open = open; });
}
document.addEventListener('DOMContentLoaded', function () {
  var ea = document.getElementById('expand-all');
  var ca = document.getElementById('collapse-all');
  if (ea) ea.addEventListener('click', function () { toggleAll(true); });
  if (ca) ca.addEventListener('click', function () { toggleAll(false); });
});
"""


def _render_html(trace: dict) -> str:
    sections = []
    for renderer in (
        _render_overview,
        _render_chunking,
        _render_schema,
        _render_extraction,
        _render_reconciliation,
        _render_answer,
    ):
        try:
            sections.append(renderer(trace))
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(f"[visualize] Section {renderer.__name__} failed: {exc}")
            sections.append(
                f'<section class="step"><p class="muted">Could not render this step ({esc(renderer.__name__)}).</p></section>'
            )

    nav = (
        '<nav class="steps">'
        '<a href="#overview">Overview</a>'
        '<a href="#chunking">1 &middot; Chunking</a>'
        '<a href="#schema">2 &middot; Schema</a>'
        '<a href="#extraction">3 &middot; Extraction</a>'
        '<a href="#reconciliation">4 &middot; Reconciliation</a>'
        '<a href="#answer">5 &middot; Query</a>'
        '<span class="spacer"></span>'
        '<button id="expand-all" type="button">Expand all</button>'
        '<button id="collapse-all" type="button">Collapse all</button>'
        "</nav>"
    )
    header = (
        '<header class="top"><h1>SLIDERS pipeline visualization</h1>'
        f'<div class="sub">Question ID: {esc(trace.get("question_id") or "run")} '
        f"&middot; generated {esc(trace.get('generated_at'))}</div></header>"
    )
    return (
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>SLIDERS report &mdash; {esc(trace.get('question_id') or 'run')}</title>"
        f"<style>{_CSS}</style></head><body>"
        f"{header}{nav}<main>{''.join(sections)}</main>"
        f"<script>{_JS}</script></body></html>"
    )


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------
def generate_visualization(
    *,
    out_dir: str | Path,
    question: str,
    question_id: str,
    documents: list,
    schema: Any,
    extracted_tables: Any,
    pre_merge_tables: Any,
    post_merge_tables: Any,
    metadata: dict,
    final_answer: Any = None,
) -> Path:
    """Build the trace, write ``trace_<qid>.json``, render ``report_<qid>.html``.

    Returns the path to the generated HTML report. Never raises for rendering
    problems -- callers should still wrap this in a try/except so a failure
    here can never break a pipeline run.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    safe_qid = str(question_id or "run").replace("/", "_").replace("\\", "_") or "run"

    trace = _build_trace(
        out_dir=out_path,
        question=question,
        question_id=question_id,
        documents=documents,
        schema=schema,
        extracted_tables=extracted_tables,
        pre_merge_tables=pre_merge_tables,
        post_merge_tables=post_merge_tables,
        metadata=metadata,
        final_answer=final_answer,
    )

    trace_path = out_path / f"trace_{safe_qid}.json"
    try:
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(trace, f, indent=2, default=str)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"[visualize] Failed to write trace JSON: {exc}")

    report_path = out_path / f"report_{safe_qid}.html"
    html_str = _render_html(trace)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_str)

    logger.info(f"[visualize] Wrote pipeline report to {report_path}")
    return report_path
