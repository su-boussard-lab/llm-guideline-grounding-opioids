# Guideline Grounding Mitigate Decision Space Framing Effect detected using Q-PAIN++

This repo audits demographic bias **and** decision-space framing effects in LLM-generated opioid dosage decisions for post-operative pain, extending the [Q-PAIN](https://www.physionet.org/content/q-pain/1.0.0/) benchmark. It shows that binary treatment-denial audits have saturated on frontier LLMs (near-100% "Yes" rates), while a large, previously-invisible instability appears once the dosage **option set** is expanded — even when the underlying clinical vignette is left completely unchanged. Guideline-grounded RAG mitigates but does not eliminate this effect.

## Key findings (see paper for full detail)
- **RQ1 — Framing effect is large:** Adding a "Medium (2 weeks)" and "None of the above" dosage option to the *unmodified* Q-PAIN vignettes drops dosage accuracy vs. ground truth by **0.21 points** (closed-book LLMs) and **0.14 points** (guideline-RAG variants), even though nothing clinical changed.
- **RQ2 — Framing is demographically uniform:** Escalation rates vary narrowly across race/gender (e.g., 30–40% for GPT-4.1-mini) with non-significant χ² tests — the effect is not a demographic bias, it's an option-set artifact.
- **RQ3/RQ4 — Persists with risk factors, doesn't propagate bias:** The same shift pattern appears at Tier 2→3 (with clinical risk factors added); KL divergence/Gini impurity across demographic subgroups stays low.
- **RQ5 — Guideline RAG helps but doesn't fix it:** Mean accuracy retention under framing rises from **0.47 → 0.85** with retrieval, but 2 of 4 models still lose accuracy when RAG-grounded.
- **RQ6 — RAG is chunk-order sensitive:** GPT-5.4's treatment-denial accuracy and BERTScore degrade under reverse-order retrieved chunks; other models are comparatively robust.
- **Sensitivity check:** Redefining "Medium" from 2→3 weeks (further from Low, closer to High) reduces uptake by only 7.2% — the shift is driven mostly by the option's *structural position* (a middle choice), not its specific clinical duration.

## Big picture: two-phase pipeline
1. **Generation** (`analysis_scripts/RAG scripts/*.py`): calls OpenAI (or Ollama for Llama-3.1-8B) over a full-factorial design (race × gender × risk factors × vignette × dosage-granularity condition) and writes raw results CSVs into `experiment_results/<model_id>/` (closed-book) or `experiment_results/<model_id>_retrieved/` (guideline-RAG variant).
2. **Analysis** (`analysis_scripts/analyze_*.py`, `visualize_*.py`, `backfill_*.py`, `normalize_*.py`, `extract_*.py`): reads those CSVs, computes bias/framing metrics, and writes tables/plots to `analysis_results/<metric_name>/<model_id>/`.

Never conflate the two — generation scripts hit paid/local model APIs and are expensive/slow; analysis scripts are pure pandas/scipy over existing CSVs and safe to rerun.

## Experimental design: Q-PAIN++ four-tier structure

| Tier | Vignette content | Dosage options | Ground-truth label? | # API calls (per model) |
|------|-------------------|-----------------|:---:|---:|
| **Tier 1** | Demographics only (unmodified Q-PAIN) | Binary (Low 1wk / High 4wk) | Yes | 10 × 8 = 80 |
| **Tier 1-DSF** | Demographics only (same unmodified vignettes as Tier 1) | 4-level (None / Low / Medium 2wk / High 4wk) | Yes | 10 × 8 = 80 |
| **Tier 2** | Demographics + risk factors | Binary (Low / High) | No | 10 × 8 × 20 = 1,600 |
| **Tier 3** | Demographics + risk factors | 4-level (None / Low / Medium / High) | No | 10 × 8 × 20 = 1,600 |

- **Demographics**: Race ∈ {Black, White, Asian, Hispanic} × Gender ∈ {man, woman} (8 subgroups), using demographic-associated first names.
- **Risk factors** (Tier 2 & 3 only, 5×2×2 = 20 combos per vignette): Mental health status (Schizophrenia, Bipolar, MDD, Anxiety, none), Opioid tolerance (Naive/Tolerant), Preoperative pain (Chronic/none).
- **Tier 1 vs. Tier 1-DSF is the core contrast**: identical vignette text, only the option set changes, so any accuracy delta is attributable purely to *decision-space framing*, not clinical content. This is the only tier pair scoreable against Q-PAIN's expert dosage annotation (`Dosage: Low` for every "Yes" vignette).
- **Sensitivity condition**: a second 4-level run with "Medium" redefined as **3 weeks** (instead of 2) to disentangle structural-position attraction from clinical-duration sensitivity.
- Filenames follow `results_[retr_evidence_]post_op_<model_id>_tier{1,2,3}_ff[_<date>].csv` — `_ff` = "full factorial"; `_retr_evidence_` prefix + `_retrieved` model folder = guideline-RAG condition.

## Models evaluated
- **GPT-5.4** (`2026-03-05`) — no log-probabilities exposed, so confidence-based metrics are omitted for this model.
- **GPT-4.1-mini** (`2025-04-14`)
- **GPT-4o-mini** (`2024-07-18`)
- **Llama-3.1-8B** — served locally via [Ollama](https://ollama.com/library/llama3.1:8b).

Each model is run in both **closed-book** and **guideline-RAG** conditions (folder suffix `_retrieved`). Inference params mirror Q-PAIN: `temperature=0`, `n=1`, `logprobs=True` where supported, `top_logprobs=5`, `max_tokens=150` (OpenAI) / `1500` (Llama-3.1, due to longer generations).

## Model_id discovery convention
Analysis scripts do NOT hardcode model lists. They use a `discover_models(experiment_results_dir)` helper (see `analyze_tier1_accuracy.py`, `analyze_rouge_l_explanations.py`) that lists subdirectories of `experiment_results/`, and a `find_tier_csv(model_dir, tier_num)` helper that globs for the right tier file inside each model folder. When adding a new analysis script, replicate this pattern rather than hardcoding paths, so it automatically picks up new models/dates.

## CSV schema conventions
Result CSVs share a wide column pattern keyed by model prefix, e.g. for `gpt4o`: `prob_gpt4o_yes`, `prob_gpt4o_no`, `prob_gpt4o_{high,low,medium,none}`, `gpt4o_answer`, `gpt4o_dosage`, `gpt4o_explanation`, `gpt4o_full`. Other columns: `vignette_idx`, `name`, `gender`, `race`, `risk_mh`, `risk_op`, `risk_pain`, `risk_text`, `closed_prompts`, `open_prompts`. Dosage values are the strings `"Low (1 week)"`, `"Medium"`, `"High"`, `"None"` — scripts map these to a numeric 0–3 scale (None=0, Low=1, Medium=2, High=3) for quantitative metrics (see `analyze_dosage_deviation.py`).

## The evaluation suite (each has its own `analyze_*.py` + `docs/*_README.md` where noted)

### Prediction-label metrics
1. **Dosage accuracy / framing-induced accuracy change** (`analyze_tier1_accuracy.py`): Tier 1 vs. Tier 1-DSF accuracy against the Q-PAIN ground-truth dosage label — the paper's primary metric. Also reports absolute 4-level accuracy and its **retention ratio** to the binary baseline (since a model already near its ceiling can't drop as far).
2. **Escalation / de-escalation rate** (`analyze_dosage_escalation.py`): unconditional decision-shift rate — did a Tier-2 "Low" baseline shift to "Medium"/"High" once Tier 3 introduces the extra options (escalation), or did a "High" baseline shift down (de-escalation, symmetric, undefined if no baseline "High" exists)? Outputs Wilson CIs, BH-FDR-adjusted p-values, χ²/Cramér's V, stratified by race/gender/opioid status.
3. **Gini impurity — individual fairness** (`analyze_gini_impurity.py`): within-vignette (Tier 2/3) disagreement across the 8 demographic subgroups for an identical clinical scenario. GI=0 means perfect agreement.
4. **KL divergence — group fairness** (`analyze_kl_divergence.py`): divergence between each demographic subgroup's dosage distribution and the population distribution across vignettes; replaces Q-PAIN's 28 pairwise Bonferroni-corrected comparisons.
5. **Dosage deviation** (`analyze_dosage_deviation.py`): within-vignette (Tier 3), subgroup mean dosage minus vignette mean dosage; t-tests vs. zero.
6. **Confidence drop** (`analyze_confidence_drop.py`): Tier1→2 and Tier2→3 change in `prob_*` of the chosen answer, with bootstrap CIs and paired permutation tests (log-probs unavailable for GPT-5.4).
7. **Standard fairness metrics** (`analyze_fairlearn_parity.py`): Demographic Parity Difference/Ratio via `fairlearn`, computed on the `predict-Low` outcome — included to show that parity-only audits stay near-uniform even while framing-induced accuracy drops substantially (a key paper argument: label-only fairness metrics are insufficient).

### Explanation-similarity metrics (compare model rationale to Q-PAIN ground-truth explanation)
8. **ROUGE-L F1** (`analyze_rouge_l_explanations.py`)
9. **BERTScore** with qwen3-8B embeddings (`analyze_rouge_l_explanations_qwen3embeddings_cosine_sim.py`)
10. **LLM-as-a-judge** clinical similarity, 1–5 scale, scored by MedGemma-27B via Ollama (`analyze_rouge_l_explanations_llm_as_a_judge.py`) — see `docs/ROUGE_L_EXPLANATION_README.md`.

### Retrieval traceability (RQ6)
11. **Chunk-order sensitivity** (`analyze_chunk_order_rouge_l.py`, `retrieved_chunk_order.py`, `run_retrieved_chunk_order_tier1.py`, `visualize_chunk_order_results.py`): re-present RAG-retrieved passages in reverse/random order and re-score treatment-denial + dosage stability, BERTScore, LLM-judge, ROUGE-L — tests whether outputs track evidence content or surface presentation order. See `docs/CHUNK_ORDER_PERMUTATION_README.md`.

Statistical helper conventions repeated across these scripts: `_wilson_ci(k, n)` for proportions, `_bh_fdr(pvals)` for multiple-comparison correction, `_two_proportion_ztest`/`_fisher_exact_pvalue` for 2×2 comparisons, `global_rate_chi2[_by_group]` for χ² + Cramér's V. Reuse these rather than reinventing stats.

## Output conventions
- All analysis outputs go under `analysis_results/<metric_name>/<model_id>/` (created relative to repo root, not `analysis_scripts/`).
- Plots always use `matplotlib.use('Agg')` before importing `pyplot` (headless-safe) — copy this pattern in new scripts.
- CSV/plot filenames end in `_ff` to signal full-factorial provenance; opioid-status-specific splits get `_opioid_naive`/`_opioid_tolerant` suffixes.

## Guideline-RAG pipeline
`analysis_scripts/RAG scripts/` and top-level `guideline-RAG/` implement retrieval-augmented generation:
- **Corpus**: 77 perioperative pain guidelines (ASA, APS, ASRA, ERAS, NICE, WHO, CDC 2022, ACS NSQIP/AGS, 2016–2025) parsed from PDF via GROBID, plus PubMed PMC Open-Access "pain management"/"perioperative care" guideline articles. A separate, broader cross-disciplinary corpus of **2,398 guidelines** (all specialties, 2016–2025) is also released for future extension beyond post-op pain (`guideline-RAG/guideline corpus/`).
- **Pipeline**: documents chunked to ~1,000 characters, embedded with `qwen3-embedding:8b` (served via Ollama), indexed in a **Chroma** vector DB. At inference time the vignette text + appended risk factors form the retrieval query; top-**K=10** chunks (cosine similarity) are inserted before the few-shot exemplars, each tagged with a provenance label (document name + section heading).
- `qpain_guideline_retrieve_rag_periop_pain.ipynb` builds/explores the retrieval index; `tier{1,2,3}_*_retr_evidence_openai.py` run the RAG-augmented generation for each tier.
- Chunk-order robustness tests (RQ6) reuse this same retrieved context but permute chunk order before re-querying the model (see Retrieval traceability above).

## Environment & running scripts
- Generation scripts load secrets via `python-dotenv` (`load_dotenv()` + `OPENAI_API_KEY` env var) — never hardcode keys. Llama-3.1-8B and qwen3-embedding/MedGemma-27B (LLM-judge) run locally via **Ollama**.
- Seeds are fixed (`np.random.seed(42)`, `random.seed(42)`) for reproducibility; preserve this when editing generation scripts.
- Dependencies: `pandas numpy matplotlib seaborn scipy` for analysis (see `docs/ANALYSIS_OVERVIEW.md`); RAG scripts additionally need `openai`, `python-dotenv`, `chromadb`; fairness metrics need `fairlearn`.
- Run analysis scripts directly, e.g. `python analysis_scripts/analyze_dosage_escalation.py`, from the repo root (paths inside scripts assume CWD = repo root, e.g. `experiment_results_dir = os.path.join(root, "experiment_results")`).
- Data-repair scripts (`backfill_experiment_dosage_from_answer.py`, `normalize_gpt54_answer_dosage.py`, `extract_llama31_dosage_from_full.py`) exist because some models' `*_dosage` column was inconsistently populated/formatted from their raw `*_full`/`*_answer` text — check these before assuming a result CSV's `_dosage` column is clean for a given model. See `docs/BACKFILL_EXPERIMENT_DOSAGE_README.md` and `docs/NORMALIZE_GPT54_ANSWER_DOSAGE_README.md`.

## Docs index
- `docs/ANALYSIS_OVERVIEW.md` — full walkthrough of all metrics, outputs, and interpretation guidance.
- `docs/GINI_IMPURITY_README.md`, `docs/KL_DIVERGENCE_README.md`, `docs/DEVIATION_ANALYSIS_README.md`, `docs/CONFIDENCE_DROP_README.md`, `docs/ESCALATION_ANALYSIS_CHANGES.md` — per-metric details.
- `docs/ROUGE_L_EXPLANATION_README.md`, `docs/CHUNK_ORDER_PERMUTATION_README.md` — explanation-similarity and retrieval-traceability details.
- `docs/BACKFILL_EXPERIMENT_DOSAGE_README.md`, `docs/NORMALIZE_GPT54_ANSWER_DOSAGE_README.md` — data-cleaning scripts for inconsistent `*_dosage` columns.

## When adding a new metric or model
1. Add a README in `docs/` mirroring existing ones (e.g. `GINI_IMPURITY_README.md`) describing purpose/definition/interpretation.
2. Follow the `discover_models` + `find_tier_csv` pattern for input discovery.
3. Write outputs to `analysis_results/<new_metric>/<model_id>/` with `_ff` suffixed filenames.
4. Reuse the shared stats helpers instead of duplicating Wilson CI / FDR / chi-square code.

## Citation
This code accompanies: Roy, S., Yu, K., & Hernandez-Boussard, T. *Beyond Binary Clinical Audits: Decision-Space Framing and the Limits of Guideline RAG in LLM Pain Management.* Building on Q-PAIN: Logé, C., et al. 2021. *Q-Pain: A question answering dataset to measure social bias in pain management.* NeurIPS 2021 Datasets and Benchmarks.
