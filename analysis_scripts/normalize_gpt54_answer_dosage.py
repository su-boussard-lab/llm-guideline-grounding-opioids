#!/usr/bin/env python3
"""
Normalize GPT-5.4 experiment CSV answer/dosage columns to canonical labels.

Targets:
  - experiment_results/gpt54/*.csv
  - experiment_results/gpt54_retrieved/*.csv

Normalization:
  *_answer -> "Yes" / "No"
  *_dosage -> "Low" / "Medium" / "High" (blank if unavailable or non-dose choice)

Makes a timestamped .bak file before modifying each CSV.
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


def _safe_str(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x)


def _pair_columns(df: pd.DataFrame) -> tuple[str, str, str | None] | None:
    dosage_cols = [c for c in df.columns if str(c).lower().endswith("_dosage")]
    if not dosage_cols:
        return None

    dc = dosage_cols[0]
    if len(dosage_cols) > 1:
        for pref in ("gpt54_dosage", "gpt4o_dosage"):
            if pref in dosage_cols:
                dc = pref
                break

    prefix = dc[: -len("_dosage")]
    ac = f"{prefix}_answer"
    fc = f"{prefix}_full"
    if ac not in df.columns:
        return None
    return dc, ac, fc if fc in df.columns else None


def _normalize_answer(answer: str, full_text: str = "") -> str:
    blob = (_safe_str(answer) + "\n" + _safe_str(full_text)).strip()
    if not blob:
        return ""

    # Prefer explicit "Answer: Yes/No" lines if present.
    for raw in blob.splitlines():
        line = raw.strip()
        m = re.match(r"^\*{0,2}\s*answer\s*:\s*\*{0,2}\s*(yes|no)\b", line, flags=re.IGNORECASE)
        if m:
            return m.group(1).capitalize()

    # Then parse initial compact style ("Yes. ...", "No - ...").
    m0 = re.match(r"^\s*(yes|no)\b", blob, flags=re.IGNORECASE)
    if m0:
        return m0.group(1).capitalize()

    # Fallback: first standalone yes/no occurrence.
    m_any = re.search(r"\b(yes|no)\b", blob, flags=re.IGNORECASE)
    if m_any:
        return m_any.group(1).capitalize()

    return ""


def _normalize_dosage(dosage: str, answer: str = "", full_text: str = "") -> str:
    blob = " ".join([_safe_str(dosage), _safe_str(answer), _safe_str(full_text)]).lower()
    if not blob:
        return ""

    # Respect user's request: only Low / Medium / High labels.
    if re.search(r"\bmedium\b", blob):
        return "Medium"
    if re.search(r"\bhigh\b", blob):
        return "High"
    if re.search(r"\blow\b", blob):
        return "Low"

    # Non-dose choices (e.g. none/n-a) remain blank.
    return ""


def normalize_csv(path: str, *, dry_run: bool) -> dict[str, int]:
    df = pd.read_csv(path)
    pair = _pair_columns(df)
    if not pair:
        return {"rows": 0, "answer_changed": 0, "dosage_changed": 0}

    dc, ac, fc = pair
    full_series = df[fc] if fc else pd.Series([""] * len(df), index=df.index)

    old_answer = df[ac].copy()
    old_dosage = df[dc].copy()

    new_answer = [
        _normalize_answer(a, f)
        for a, f in zip(df[ac], full_series)
    ]
    new_dosage = [
        _normalize_dosage(d, a, f)
        for d, a, f in zip(df[dc], df[ac], full_series)
    ]

    answer_changed = int((old_answer.astype(str).fillna("") != pd.Series(new_answer, index=df.index).astype(str).fillna("")).sum())
    dosage_changed = int((old_dosage.astype(str).fillna("") != pd.Series(new_dosage, index=df.index).astype(str).fillna("")).sum())

    if (answer_changed or dosage_changed) and not dry_run:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        bak = path + f".bak.{ts}"
        shutil.copy2(path, bak)
        df[ac] = new_answer
        df[dc] = new_dosage
        df.to_csv(path, index=False)

    return {
        "rows": int(len(df)),
        "answer_changed": answer_changed,
        "dosage_changed": dosage_changed,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dirs",
        nargs="+",
        default=[
            os.path.join("experiment_results", "gpt54"),
            os.path.join("experiment_results", "gpt54_retrieved"),
        ],
    )
    ap.add_argument("--glob", default="*.csv")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(root)

    total_files = total_rows = total_a = total_d = 0
    for d in args.dirs:
        dpath = d if os.path.isabs(d) else os.path.join(root, d)
        if not os.path.isdir(dpath):
            print(f"SKIP (not found): {dpath}")
            continue
        for path in sorted(glob.glob(os.path.join(dpath, args.glob))):
            if not path.endswith(".csv"):
                continue
            stats = normalize_csv(path, dry_run=args.dry_run)
            if stats["rows"] == 0:
                continue
            total_files += 1
            total_rows += stats["rows"]
            total_a += stats["answer_changed"]
            total_d += stats["dosage_changed"]
            rel = os.path.relpath(path, root)
            tag = " [dry-run]" if args.dry_run else ""
            print(
                f"{rel}: rows={stats['rows']}, answer_changed={stats['answer_changed']}, "
                f"dosage_changed={stats['dosage_changed']}{tag}"
            )

    tag = " [dry-run]" if args.dry_run else ""
    print(
        f"\nProcessed files={total_files}, rows={total_rows}, "
        f"answer_changed={total_a}, dosage_changed={total_d}{tag}"
    )


if __name__ == "__main__":
    main()
