#!/usr/bin/env python3
"""Quick sanity check: parse the cached OA file list and show matches."""
import csv
import re
import sys
from pathlib import Path

csv_path = Path(__file__).resolve().parent.parent / "cache" / "oa_file_list.csv"
if not csv_path.exists():
    print(f"ERROR: {csv_path} not found. Run step 1 first to download it.")
    sys.exit(1)

year_pattern = re.compile(r"\b(19\d{2}|20\d{2})\b")
count = 0

with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    print(f"CSV columns: {reader.fieldnames}\n")

    for row in reader:
        citation = row.get("Article Citation", "")
        m = year_pattern.search(citation)
        if m and int(m.group(1)) >= 2000:
            count += 1
            if count <= 5:
                year = m.group(1)
                journal = citation[: m.start()].rstrip(". ")
                pmcid = row.get("Accession ID", "?")
                fpath = row.get("File", "?")
                print(f"  [{count}] year={year}  pmcid={pmcid}  journal={journal!r}")
                print(f"       file={fpath}")
        if count >= 10000:
            break

print(f"\nTotal papers with year >= 2000 (scanned up to 10k): {count}")
if count == 0:
    print("WARNING: No papers matched! Check the CSV format.")
else:
    print("Parsing is working correctly.")

