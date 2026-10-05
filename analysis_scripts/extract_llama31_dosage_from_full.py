#!/usr/bin/env python3
"""
Backfill empty gpt4o_dosage cells in Llama 3.1 tier CSVs from gpt4o_full (and gpt4o_answer).

Llama 3.1 closed-book runs often leave gpt4o_dosage empty while gpt4o_full contains the
structured reply (including multi-patient batches keyed by patient name).

Default targets:
  experiment_results/llama31/results_post_op_llama3_1_tier2_ff_May23.csv
  experiment_results/llama31/results_post_op_llama3_1_tier3_ff_May19.csv

Usage (from Q-Pain root):
  python analysis_scripts/extract_llama31_dosage_from_full.py
  python analysis_scripts/extract_llama31_dosage_from_full.py --dry-run
  python analysis_scripts/extract_llama31_dosage_from_full.py --csv path/to/tier3.csv
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from datetime import datetime, timezone

import numpy as np
import pandas as pd

DEFAULT_CSVS = [
    os.path.join(
        "experiment_results",
        "llama31",
        "results_post_op_llama3_1_tier2_ff_May23.csv",
    ),
    os.path.join(
        "experiment_results",
        "llama31",
        "results_post_op_llama3_1_tier3_ff_May19.csv",
    ),
]

ROMAN_TO_DOSAGE = {
    "i": "Low (1 week)",
    "ii": "Medium (2 weeks)",
    "iii": "High (4 weeks)",
    "iv": "None of the above",
}

NAN_DOSAGE_VALUES = frozenset(
    {
        "nan",
        "n/a",
        "na",
        "not applicable",
        "none",
        "no supply",
    }
)


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _is_empty_dosage(value) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return True
    text = str(value).strip()
    return text == "" or text.lower() == "nan"


def _safe_str(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value)


def _normalize_dosage(raw: str) -> str:
    """Map extracted text to canonical experiment dosage strings."""
    text = raw.strip().rstrip(".")
    if not text:
        return text
    low = text.lower()
    if low in NAN_DOSAGE_VALUES or "not applicable" in low:
        return "nan (not applicable)"
    if re.search(r"none\s+of\s+the\s+above", low):
        return "None of the above"
    if re.search(r"medium\s*\(\s*2\s*weeks?\s*\)", low):
        return "Medium (2 weeks)"
    if re.search(r"high\s*\(\s*4\s*weeks?\s*\)", low):
        return "High (4 weeks)"
    if re.search(r"low\s*\(\s*1\s*week\s*\)", low):
        return "Low (1 week)"
    # Tier 1-style if ever present
    if re.search(r"low\s*\(\s*0\.5\s*mg\s*\)", low):
        return "Low (0.5 mg)"
    if re.search(r"high\s*\(\s*1\s*mg\s*\)", low):
        return "High (1 mg)"
    return text


def _section_for_name(full_text: str, name: str) -> str:
    """
    Return the substring of full_text most likely describing this patient.

    Handles **Name**: blocks, Name: lines, and single-patient preambles
    ('... answer the question for Name.').
    """
    full_text = _safe_str(full_text)
    name = _safe_str(name).strip()
    if not full_text:
        return ""
    if not name:
        return full_text

    escaped = re.escape(name)
    block_patterns = [
        rf"\*\*{escaped}(?:\s*\([^)]*\))?\*\*:?(.*?)(?=\n\*\*[^*]+\*\*|\Z)",
        rf"\*\*{escaped}(?:\s*\([^)]*\))?\*\*(.*?)(?=\n\*\*[^*]+\*\*|\Z)",
        rf"(?:^|\n){escaped}(?:\s*\([^)]*\))?:\s*(.*?)(?=\n(?:[A-Z][a-z]+|\*\*)|\Z)",
    ]
    for pattern in block_patterns:
        match = re.search(pattern, full_text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(0)

    if re.search(rf"\bfor\s+{escaped}\b", full_text, flags=re.IGNORECASE):
        return full_text

    return full_text


def _dosage_from_dosage_line(blob: str) -> str | None:
    """Parse 'Dosage: ...' (line or inline before Explanation)."""
    for raw in blob.splitlines():
        line = raw.strip()
        match = re.match(
            r"^dosage\s*:\s*(.+?)(?:\.\s*(?:explanation|$)|\s*explanation\s*:|$)",
            line,
            flags=re.IGNORECASE,
        )
        if match:
            value = match.group(1).strip().rstrip(".")
            if value and value.lower() not in NAN_DOSAGE_VALUES:
                return value

    inline = re.search(
        r"dosage\s*:\s*([^.\n]+(?:\([^)]+\))?)",
        blob,
        flags=re.IGNORECASE,
    )
    if inline:
        value = inline.group(1).strip().rstrip(".")
        if value.lower() not in NAN_DOSAGE_VALUES:
            return value
    return None


def _dosage_from_answer_line(blob: str) -> str | None:
    """Parse 'Answer: (iii) High (4 weeks)' or roman-only answers."""
    match = re.search(
        r"answer\s*:\s*\(?([ivx]+)\)?\s*([^.\n]*)",
        blob,
        flags=re.IGNORECASE,
    )
    if match:
        roman = match.group(1).lower()
        rest = match.group(2).strip().rstrip(".")
        if rest:
            return rest
        return ROMAN_TO_DOSAGE.get(roman)

    plain = re.search(r"answer\s*:\s*(.+?)(?:\n|$)", blob, flags=re.IGNORECASE)
    if plain:
        value = plain.group(1).strip().rstrip(".")
        if value:
            return value
    return None


def _dosage_from_compact_text(text: str) -> str | None:
    """Infer dosage from short answer lines or roman-choice-only replies."""
    blob = _safe_str(text).strip()
    if not blob:
        return None

    low = blob.lower()
    if re.search(r"none\s+of\s+the\s+above", low):
        return "None of the above"
    if re.search(r"medium\s*\(\s*2\s*weeks?\s*\)", low):
        return "Medium (2 weeks)"
    if re.search(r"high\s*\(\s*4\s*weeks?\s*\)", low):
        return "High (4 weeks)"
    if re.search(r"low\s*\(\s*1\s*week\s*\)", low):
        return "Low (1 week)"

    roman_with_text = re.match(r"^\(([ivx]+)\)\s*(.+)$", blob, flags=re.IGNORECASE)
    if roman_with_text:
        roman = roman_with_text.group(1).lower()
        rest = roman_with_text.group(2).strip()
        return rest or ROMAN_TO_DOSAGE.get(roman)

    roman_only = re.match(r"^\(([ivx]+)\)$", blob, flags=re.IGNORECASE)
    if roman_only:
        return ROMAN_TO_DOSAGE.get(roman_only.group(1).lower())

    return None


def extract_dosage_from_full(name: str, answer: str, full_text: str) -> str | None:
    """
    Extract a dosage string for one vignette row.

    Search order: patient-specific section of full_text, entire full_text, answer.
    """
    candidates = (
        _section_for_name(full_text, name),
        _safe_str(full_text),
        _safe_str(answer),
    )
    for blob in candidates:
        if not blob or blob.lower() == "nan":
            continue
        for parser in (_dosage_from_dosage_line, _dosage_from_answer_line, _dosage_from_compact_text):
            found = parser(blob)
            if found:
                normalized = _normalize_dosage(found)
                if normalized:
                    return normalized
    return None


def backfill_csv(path: str, *, dry_run: bool, report_failures: str | None) -> tuple[int, int, int]:
    """
    Fill empty gpt4o_dosage from gpt4o_full / gpt4o_answer.

    Returns (rows_needing_fill, rows_filled, rows_still_empty).
    """
    df = pd.read_csv(path)
    required = {"name", "gpt4o_dosage", "gpt4o_full"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")

    answer_col = "gpt4o_answer" if "gpt4o_answer" in df.columns else None
    need_mask = df["gpt4o_dosage"].apply(_is_empty_dosage)
    n_need = int(need_mask.sum())
    if n_need == 0:
        return 0, 0, 0

    filled = 0
    still_empty_indices: list[int] = []
    new_dosage = df["gpt4o_dosage"].copy()

    for idx in df.index[need_mask]:
        answer = df.at[idx, answer_col] if answer_col else ""
        extracted = extract_dosage_from_full(
            name=df.at[idx, "name"],
            answer=answer,
            full_text=df.at[idx, "gpt4o_full"],
        )
        if extracted is not None:
            new_dosage.at[idx] = extracted
            filled += 1
        else:
            still_empty_indices.append(idx)

    n_still = n_need - filled

    if report_failures and still_empty_indices:
        fail_df = df.loc[still_empty_indices, ["vignette_idx", "name", "gpt4o_answer", "gpt4o_full"]].copy()
        os.makedirs(os.path.dirname(report_failures) or ".", exist_ok=True)
        fail_df.to_csv(report_failures, index=False)

    if filled == 0:
        return n_need, 0, n_still

    if not dry_run:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_path = f"{path}.bak.{timestamp}"
        shutil.copy2(path, backup_path)
        df["gpt4o_dosage"] = new_dosage
        df.to_csv(path, index=False)
        print(f"  backup: {backup_path}")

    return n_need, filled, n_still


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        nargs="+",
        default=DEFAULT_CSVS,
        help="Tier CSV path(s) to update (default: llama31 tier2 May23 + tier3 May19).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts only; do not write files.",
    )
    parser.add_argument(
        "--report-failures",
        metavar="DIR",
        default=None,
        help="If set, write <basename>_unfilled.csv per input file into this directory.",
    )
    args = parser.parse_args()

    root = _project_root()
    os.chdir(root)

    total_need = total_filled = total_still = 0
    for csv_arg in args.csv:
        path = csv_arg if os.path.isabs(csv_arg) else os.path.join(root, csv_arg)
        if not os.path.isfile(path):
            print(f"SKIP (not found): {path}")
            continue

        report_path = None
        if args.report_failures:
            base = os.path.splitext(os.path.basename(path))[0]
            report_path = os.path.join(args.report_failures, f"{base}_unfilled.csv")

        need, filled, still = backfill_csv(
            path,
            dry_run=args.dry_run,
            report_failures=report_path,
        )
        rel = os.path.relpath(path, root)
        suffix = " [dry-run]" if args.dry_run else ""
        print(
            f"{rel}: empty dosage={need}, filled={filled}, still empty={still}{suffix}"
        )
        if report_path and still:
            print(f"  failures report: {report_path}")

        total_need += need
        total_filled += filled
        total_still += still

    suffix = " [dry-run]" if args.dry_run else ""
    print(
        f"\nTotal: empty={total_need}, filled={total_filled}, still empty={total_still}{suffix}"
    )


if __name__ == "__main__":
    main()
