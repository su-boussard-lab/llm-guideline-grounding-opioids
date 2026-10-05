#!/usr/bin/env python3
"""
Re-run Tier 1 retrieved experiments with permuted RAG chunk order.

Uses existing Tier 1 retrieved CSVs as templates (same closed prompt, same 10 chunks,
reordered open_prompt context). Runs two permutations by default:
  - reverse: reverse chunk list per vignette
  - random:  seeded shuffle per vignette_idx (shared across 8 demographics)

Baseline (original order) is NOT re-run; compare to the source Tier 1 CSV via ROUGE-L.

Usage (from Q-Pain root):
  python analysis_scripts/run_retrieved_chunk_order_tier1.py --dry-run
  python analysis_scripts/run_retrieved_chunk_order_tier1.py --models gpt54_retrieved
  python analysis_scripts/run_retrieved_chunk_order_tier1.py --limit 2
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from retrieved_chunk_order import apply_chunk_permutation

PERMUTATIONS = ("reverse", "random")

# Folder name -> OpenAI model id (override with env OPENAI_MODEL for gpt54_retrieved)
MODEL_ID_BY_FOLDER: dict[str, str] = {
    "gpt54_retrieved": os.getenv("OPENAI_MODEL", "gpt-5.4"),
    "gpt4o_mini_retrieved": "gpt-4o-mini",
    "gpt41_mini_retrieved": "gpt-4.1-mini",
}


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def discover_retrieved_models(experiment_results_dir: str) -> list[str]:
    if not os.path.isdir(experiment_results_dir):
        return []
    out: list[str] = []
    for name in sorted(os.listdir(experiment_results_dir)):
        if not name.endswith("_retrieved"):
            continue
        p = os.path.join(experiment_results_dir, name)
        if os.path.isdir(p) and not name.startswith("."):
            out.append(name)
    return out


def find_tier1_csv(model_dir: str) -> str | None:
    patterns = [
        os.path.join(model_dir, "*tier1*_ff*.csv"),
        os.path.join(model_dir, "*tier1*.csv"),
    ]
    for pat in patterns:
        matches = [
            m
            for m in sorted(glob.glob(pat))
            if ".bak." not in os.path.basename(m)
        ]
        if matches:
            return matches[0]
    return None


def _detect_output_prefix(df: pd.DataFrame) -> str:
    for c in df.columns:
        if str(c).lower().endswith("_answer"):
            return str(c)[: -len("_answer")]
    return "gpt4o"


def _parse_model_output(gen_text: str) -> tuple[str, str, str]:
    if not gen_text:
        return "", "", ""
    lines = gen_text.strip().split("\n")
    answer, dosage, explanation = "", "", ""
    for line in lines:
        if line.startswith("Answer:"):
            answer = line.replace("Answer:", "", 1).strip()
        elif line.startswith("Dosage:"):
            dosage = line.replace("Dosage:", "", 1).strip()
        elif line.startswith("Explanation:"):
            explanation = line.replace("Explanation:", "", 1).strip()
    if not answer and lines:
        answer = lines[0].strip()
    if not dosage and len(lines) > 1:
        dosage = lines[1].strip()
    return answer, dosage, explanation


def _call_openai(
    client: OpenAI,
    *,
    model_id: str,
    final_prompt: str,
    temperature: float,
    max_completion_tokens: int,
    max_retries: int = 5,
) -> str:
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": final_prompt}],
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
            )
            text = (resp.choices[0].message.content or "").strip()
            if "##" in text:
                text = text.split("##", 1)[0].strip()
            return text
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                sleep_time = 2 * (2**attempt)
                print(f"    API error (attempt {attempt + 1}/{max_retries}): {e} — sleeping {sleep_time}s")
                time.sleep(sleep_time)
            else:
                raise
    raise RuntimeError(str(last_err))


def _row_key(row: pd.Series) -> tuple:
    return (
        int(row["vignette_idx"]),
        str(row["race"]),
        str(row["gender"]),
        str(row.get("name", "")),
    )


def _load_done_keys(out_path: str) -> set[tuple]:
    if not os.path.isfile(out_path):
        return set()
    done = pd.read_csv(out_path)
    if "permutation_id" not in done.columns:
        return set()
    keys: set[tuple] = set()
    for _, r in done.iterrows():
        keys.add((str(r["permutation_id"]), *_row_key(r)))
    return keys


def run_model(
    *,
    model_folder: str,
    source_csv: str,
    out_path: str,
    client: OpenAI | None,
    model_id: str,
    permutations: tuple[str, ...],
    random_seed: int,
    temperature: float,
    max_completion_tokens: int,
    sleep_s: float,
    dry_run: bool,
    limit: int | None,
    resume: bool,
) -> int:
    df = pd.read_csv(source_csv)
    required = {"vignette_idx", "race", "gender", "closed_prompts", "open_prompts"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{model_folder}: missing columns {sorted(missing)}")

    prefix = _detect_output_prefix(df)
    ans_col, dos_col, expl_col, full_col = (
        f"{prefix}_answer",
        f"{prefix}_dosage",
        f"{prefix}_explanation",
        f"{prefix}_full",
    )

    done_keys: set[tuple] = _load_done_keys(out_path) if resume else set()
    n_calls = 0
    rows_out: list[dict[str, Any]] = []

    work = df.head(limit) if limit is not None else df

    for perm in permutations:
        print(f"  Permutation: {perm}")
        for idx, row in work.iterrows():
            key = (perm, *_row_key(row))
            if key in done_keys:
                continue

            v_idx = int(row["vignette_idx"])
            try:
                new_open, orig_order, perm_order = apply_chunk_permutation(
                    str(row["open_prompts"]),
                    perm,  # type: ignore[arg-type]
                    vignette_idx=v_idx,
                    random_seed=random_seed,
                )
            except Exception as e:
                print(f"    SKIP row {idx} (v{v_idx} {row['race']} {row['gender']}): parse error {e}")
                continue

            final_prompt = str(row["closed_prompts"]) + new_open
            gen_text = ""
            if not dry_run:
                if client is None:
                    raise RuntimeError("OpenAI client required when not in --dry-run mode")
                gen_text = _call_openai(
                    client,
                    model_id=model_id,
                    final_prompt=final_prompt,
                    temperature=temperature,
                    max_completion_tokens=max_completion_tokens,
                )
                if sleep_s > 0:
                    time.sleep(sleep_s)

            answer, dosage, explanation = _parse_model_output(gen_text)

            out_row: dict[str, Any] = {
                "context": row.get("context", "Postoperative Pain"),
                "vignette_idx": v_idx,
                "name": row.get("name", ""),
                "gender": row["gender"],
                "race": row["race"],
                "permutation_id": perm,
                "baseline_chunk_order": orig_order,
                "chunk_order": perm_order,
                "source_model_folder": model_folder,
                "source_tier1_csv": os.path.basename(source_csv),
                ans_col: answer,
                dos_col: dosage,
                expl_col: explanation,
                full_col: gen_text,
                "closed_prompts": row["closed_prompts"],
                "open_prompts": new_open,
            }
            # Preserve prob_* columns if present (not re-computed)
            for c in df.columns:
                if c.startswith("prob_") and c not in out_row:
                    out_row[c] = row[c]

            rows_out.append(out_row)
            n_calls += 1

            if n_calls % 10 == 0:
                print(f"    completed {n_calls} new calls ({perm})")

    if rows_out and not dry_run:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        new_df = pd.DataFrame(rows_out)
        if os.path.isfile(out_path):
            existing = pd.read_csv(out_path)
            new_df = pd.concat([existing, new_df], ignore_index=True)
        new_df.to_csv(out_path, index=False)

    return n_calls


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Retrieved model folders under experiment_results/ (default: all *_retrieved)",
    )
    ap.add_argument(
        "--permutations",
        nargs="*",
        choices=list(PERMUTATIONS),
        default=list(PERMUTATIONS),
        help="Which permutations to run (default: reverse random)",
    )
    ap.add_argument("--random-seed", type=int, default=42, help="Base seed for random permutation")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-completion-tokens", type=int, default=1200)
    ap.add_argument("--sleep", type=float, default=1.0, help="Seconds between API calls")
    ap.add_argument("--dry-run", action="store_true", help="Build prompts only; no API calls")
    ap.add_argument("--limit", type=int, default=None, help="Max rows per model (for testing)")
    ap.add_argument("--no-resume", action="store_true", help="Do not skip rows already in output CSV")
    ap.add_argument(
        "--output-subdir",
        default="chunk_order_tier1",
        help="Subfolder under each model dir for outputs",
    )
    args = ap.parse_args()

    root = _project_root()
    os.chdir(root)
    load_dotenv(os.path.join(root, ".env"))

    exp_dir = os.path.join(root, "experiment_results")
    models = args.models or discover_retrieved_models(exp_dir)
    if not models:
        print("No *_retrieved model folders found under experiment_results/")
        return

    client: OpenAI | None = None
    if not args.dry_run:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("OPENAI_API_KEY not set in environment / .env")
        client = OpenAI(api_key=api_key)

    total_calls = 0
    print(f"Permutations: {args.permutations}")
    print(f"Random seed base: {args.random_seed}")
    if args.dry_run:
        print("DRY RUN — no API calls")

    for model_folder in models:
        model_dir = os.path.join(exp_dir, model_folder)
        source_csv = find_tier1_csv(model_dir)
        if not source_csv:
            print(f"SKIP {model_folder}: no Tier 1 CSV")
            continue

        model_id = MODEL_ID_BY_FOLDER.get(model_folder)
        if not model_id:
            print(f"SKIP {model_folder}: no model id mapping (add to MODEL_ID_BY_FOLDER)")
            continue

        out_dir = os.path.join(model_dir, args.output_subdir)
        out_path = os.path.join(out_dir, "results_tier1_chunk_order_permutations_ff.csv")

        n_source = len(pd.read_csv(source_csv))
        expected = n_source * len(args.permutations)
        print(f"\n{'=' * 72}")
        print(f"MODEL FOLDER: {model_folder}")
        print(f"  OpenAI model: {model_id}")
        print(f"  Source: {source_csv} ({n_source} rows)")
        print(f"  Output: {out_path}")
        print(f"  New API calls (max): {expected}")

        n = run_model(
            model_folder=model_folder,
            source_csv=source_csv,
            out_path=out_path,
            client=client,
            model_id=model_id,
            permutations=tuple(args.permutations),
            random_seed=args.random_seed,
            temperature=args.temperature,
            max_completion_tokens=args.max_completion_tokens,
            sleep_s=args.sleep,
            dry_run=args.dry_run,
            limit=args.limit,
            resume=not args.no_resume,
        )
        total_calls += n
        print(f"  Done: {n} new calls this run")

    print(f"\nTotal new API calls this run: {total_calls}")
    if not args.dry_run and total_calls:
        print("Compare explanations to baseline Tier 1 CSV with analyze_rouge_l_explanations.py")


if __name__ == "__main__":
    main()
