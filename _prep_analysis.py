import json
import os
import csv

ROOT = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(ROOT, "datasets/financebench/financebench_open_source.jsonl")
IDS_CSV = os.path.join(ROOT, "sample_data/30_sample_ids_financebench_ablation.csv")
MD_DIR = os.path.join(ROOT, "datasets/financebench/markdown")
PDF_DIR = os.path.join(ROOT, "datasets/financebench/pdfs")

# Load 30 specific ids
ids = []
with open(IDS_CSV) as f:
    for row in csv.reader(f):
        if not row:
            continue
        v = row[0].strip()
        if v.startswith("#") or v == "id" or not v:
            continue
        ids.append(v)
ids = set(ids)

# Load benchmark, build id -> doc_name
id_to_doc = {}
all_docs = set()
with open(BENCH) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        id_to_doc[obj["financebench_id"]] = obj["doc_name"]
        all_docs.add(obj["doc_name"])

# doc_names needed for the 30 ids
needed_docs = {}
missing_ids = []
for i in sorted(ids):
    doc = id_to_doc.get(i)
    if doc is None:
        missing_ids.append(i)
    else:
        needed_docs.setdefault(doc, []).append(i)

existing_md = {f[:-3] for f in os.listdir(MD_DIR) if f.endswith(".md")}
existing_pdf = {f[:-4] for f in os.listdir(PDF_DIR) if f.endswith(".pdf")}

print(f"# of selected ids: {len(ids)}")
print(f"# ids not found in benchmark: {len(missing_ids)} -> {missing_ids}")
print(f"# unique docs needed for the 30 ids: {len(needed_docs)}")
print()
print("=== Per needed doc: has .md ? has .pdf ? ===")
md_present = []
md_missing_pdf_present = []
md_missing_pdf_missing = []
for doc in sorted(needed_docs):
    has_md = doc in existing_md
    has_pdf = doc in existing_pdf
    flag = "OK" if has_md else ("NEED-CONVERT" if has_pdf else "MISSING-PDF!")
    print(f"  [{flag:13}] {doc}  (md={has_md}, pdf={has_pdf})  ids={needed_docs[doc]}")
    if has_md:
        md_present.append(doc)
    elif has_pdf:
        md_missing_pdf_present.append(doc)
    else:
        md_missing_pdf_missing.append(doc)

print()
print(f"Docs with markdown ready:           {len(md_present)}")
print(f"Docs needing PDF->MD conversion:    {len(md_missing_pdf_present)} -> {md_missing_pdf_present}")
print(f"Docs missing BOTH md and pdf:       {len(md_missing_pdf_missing)} -> {md_missing_pdf_missing}")

print()
print("=== Existing markdown files ===")
for m in sorted(existing_md):
    print(f"  {m}")
print()
print(f"Total PDFs available: {len(existing_pdf)}")
