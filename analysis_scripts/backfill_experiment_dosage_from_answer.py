#!/usr/bin/env python3
"""
Backfill empty *_dosage cells from *_answer and/or *_full in experiment CSVs.

Some GPT-5.x runs put the full reply in the answer field (e.g. "Yes. Low (1 week) supply.")
without a separate structured "Dosage:" line parsed into *_dosage, which breaks downstream
analyses (e.g. Gini heatmaps showing '?').

Usage:
  python analysis_scripts/backfill_experiment_dosage_from_answer.py \\
    --dirs experiment_results/gpt54 experiment_results/gpt54_retrieved

  # Preview only:
  python analysis_scripts/backfill_experiment_dosage_from_answer.py --dirs ... --dry-run

By default writes in place and saves a timestamped .bak copy next to each modified CSV.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
from datetime import datetime, timezone

import numpy as np
import pandas as pd


def _is_empty_dosage(s) -> bool:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return True
    t = str(s).strip()
    return t == "" or t.lower() == "nan"


def _safe_str(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x)


def _dosage_line_from_blob(blob: str) -> str | None:
    """First 'Dosage: ...' line value (anywhere in blob)."""
    for raw in blob.splitlines():
        line = raw.strip()
        m = re.match(r"^dosage\s*:\s*(.+)$", line, flags=re.IGNORECASE)
        if m:
            val = m.group(1).strip()
            if val:
                return val
    return None


def _infer_dosage_from_compact_answer(text: str) -> str | None:
    """
    Parse dosage embedded in a single-line style answer, e.g.
    'Yes. Low (1 week) supply.' / 'Yes — Low (1 week) supply.' / 'Yes — Medium (2 weeks).'
    Order: longest / most specific first.
    """
    t = _safe_str(text)
    if not t:
        return None
    low = t.lower()

    # Tier 3 / four-way wording
    if re.search(r"none\s+of\s+the\s+above", low):
        return "None of the above"
    if re.search(r"medium\s*\(\s*2\s*weeks?\s*\)", low):
        return "Medium (2 weeks)"
    if re.search(r"high\s*\(\s*4\s*weeks?\s*\)", low):
        return "High (4 weeks)"
    if re.search(r"low\s*\(\s*1\s*week\s*\)", low):
        return "Low (1 week)"

    # Tier 1 style (mg) if present in text
    if re.search(r"low\s*\(\s*0\.5\s*mg\s*\)", low):
        return "Low (0.5 mg)"
    if re.search(r"high\s*\(\s*1\s*mg\s*\)", low):
        return "High (1 mg)"

    return None


def extract_dosage_for_row(answer: str, full_text: str) -> str | None:
    """Return a non-empty dosage string to write, or None if unknown."""
    blob = _safe_str(full_text) + "\n" + _safe_str(answer)
    from_line = _dosage_line_from_blob(blob)
    if from_line and not _is_empty_dosage(from_line):
        return from_line.strip()
    inferred = _infer_dosage_from_compact_answer(_safe_str(answer))
    if inferred:
        return inferred
    # Sometimes only full has compact form without "Dosage:" prefix
    inferred2 = _infer_dosage_from_compact_answer(_safe_str(full_text))
    if inferred2:
        return inferred2
    return None


def _pair_columns(df: pd.DataFrame) -> tuple[str, str, str | None] | None:
    """Return (dosage_col, answer_col, full_col_or_None) for the first *_dosage prefix."""
    dosage_cols = [c for c in df.columns if str(c).lower().endswith("_dosage")]
    if not dosage_cols:
        return None
    dc = dosage_cols[0]
    if len(dosage_cols) > 1:
        # Prefer gpt54_ then gpt4o_
        for pref in ("gpt54_dosage", "gpt4o_dosage"):
            if pref in dosage_cols:
                dc = pref
                break
    prefix = dc[: -len("_dosage")]
    ac = f"{prefix}_answer"
    fc = f"{prefix}_full"
    if ac not in df.columns:
        return None
    full_col = fc if fc in df.columns else None
    return dc, ac, full_col


def backfill_csv(path: str, *, dry_run: bool) -> tuple[int, int]:
    """
    Returns (n_rows_needing_fill, n_rows_filled).
    """
    df = pd.read_csv(path)
    pair = _pair_columns(df)
    if not pair:
        return 0, 0
    dc, ac, fc = pair

    need = df[dc].apply(_is_empty_dosage)
    n_need = int(need.sum())
    if n_need == 0:
        return 0, 0

    filled = 0
    new_vals = df[dc].copy()
    for idx in df.index[need]:
        ans = df.at[idx, ac]
        full = df.at[idx, fc] if fc else ""
        got = extract_dosage_for_row(ans, full)
        if got is not None:
            new_vals.at[idx] = got
            filled += 1

    if filled == 0:
        return n_need, 0

    if not dry_run:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        bak = path + f".bak.{ts}"
        shutil.copy2(path, bak)
        df[dc] = new_vals
        df.to_csv(path, index=False)

    return n_need, filled


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dirs",
        nargs="+",
        default=[
            os.path.join("experiment_results", "gpt54"),
            os.path.join("experiment_results", "gpt54_retrieved"),
        ],
        help="Directories containing tier CSVs to patch.",
    )
    ap.add_argument(
        "--glob",
        default="*.csv",
        help="Filename glob within each dir (default: all CSV).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would change; do not write files.",
    )
    args = ap.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(root)

    total_need = total_filled = 0
    for d in args.dirs:
        dpath = d if os.path.isabs(d) else os.path.join(root, d)
        if not os.path.isdir(dpath):
            print(f"SKIP (not a dir): {dpath}")
            continue
        pattern = os.path.join(dpath, args.glob)
        for path in sorted(glob.glob(pattern)):
            if not path.endswith(".csv"):
                continue
            need, filled = backfill_csv(path, dry_run=args.dry_run)
            if need:
                rel = os.path.relpath(path, root)
                print(f"{rel}: rows with empty dosage={need}, backfilled={filled}" + (" [dry-run]" if args.dry_run else ""))
                total_need += need
                total_filled += filled

    print(f"\nTotal: empty dosage rows seen={total_need}, backfilled={total_filled}" + (" [dry-run]" if args.dry_run else ""))


if __name__ == "__main__":
    main()
