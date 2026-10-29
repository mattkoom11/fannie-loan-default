# Fannie Mae Loan Default Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Predict 24-month serious default for Fannie Mae single-family mortgages from origination data. Compare logistic regression, random forest and gradient boosting on an out-of-time test set, choose a cost-based decision threshold, and report default rates by origination vintage.

**Architecture:** A small `fannie_default` package does the reusable work. It streams the pipe-delimited quarterly files in chunks, samples loans deterministically by ID, builds one row per loan (origination features plus a 24-month default label), and holds the preprocessing, models and evaluation functions. Two notebooks tell the story: vintage analysis, then the model comparison. They import the package and contain no logic of their own, so everything they show is tested.

**Tech Stack:** Python 3.12 (uv-managed), pandas, pyarrow, scikit-learn, LightGBM, matplotlib, Jupyter, pytest.

**Spec:** `~/Documents/GitHub/Juwon-s-Vault/wiki/concepts/jobs/resume-skills-backlog.md`: "Top priorities" item 1 and "Early-learning order". No separate design doc. Matthew skipped the design Q&A on 2026-10-28, so every choice below not stated in that page is the plan author's call. Those calls are listed in [Decisions made by the plan](#decisions-made-by-the-plan).

## Global Constraints

- Data: Fannie Mae Single-Family Loan Performance Data, combined quarterly files (`YYYYQn.csv`: pipe-delimited, no header). Downloading needs a free Data Dynamics account (capitalmarkets.fanniemae.com).
- Raw and processed data are **never committed**; the terms of use forbid redistributing them. `data/` is gitignored. The test fixtures are synthetic.
- Use a subset, not the full files: the default loan sample rate is `0.10`.
- Origination vintages: `2015Q1`–`2017Q4`. Train on `2015Q1`–`2016Q4`; the out-of-time test set is `2017Q1`–`2017Q4`.
- The label `default_24m` is 1 if, within the first 24 months after the first payment date, the loan is ever 180+ days delinquent (`DLQ_STATUS >= 6`; changed from 90+ on 2026-10-28 by Matthew, after the first real build showed 2017 hurricane forbearance inflating 90-day delinquency in FL/TX) or ends with zero-balance code `02`, `03`, `09` or `15`.
- Features come from **origination fields only**. No performance-period field may become a feature.
- Random seed `42` everywhere.
- Models: logistic regression baseline, then random forest, then LightGBM. Compare them on ROC AUC and PR AUC, plus a cost-based threshold.
- **Learning gates (Matthew):** each model is built only after Matthew has had a `teach` session on its method (the backlog's bar: learned, used, explainable). The agent stops at each gate.

## Review Focus

1. **Dates read as integers.** `ACT_PERIOD`/`ORIG_DATE`/`FIRST_PAY` are `MMYYYY`, so `012016` becomes `12016` if parsed as an int. They must load as strings and parse to month periods (pinned in Task 2).
2. **A loan's rows split across chunks.** The label has to be computed after all of a sampled loan's rows are collected, not chunk by chunk. Hash sampling keeps every row of a sampled loan (pinned in Tasks 2 and 3).
3. **Unknown delinquency status (`XX`, blank).** This must count as missing, not as current (0) and not as an error (pinned in Task 3).
4. **Right-censored loans.** A loan seen for less than 24 months that didn't terminate must be dropped, not labeled 0 (pinned in Task 3).
5. **Missing credit score or DTI.** A blank `CSCORE_B` or `DTI` must be imputed with a missing indicator and must not crash the pipeline or the logistic model (pinned in Task 5).

---

## File Structure

```
fannie-loan-default/
  pyproject.toml                 uv project, deps, pytest config
  .gitignore                     data/, .venv/, .ipynb_checkpoints/
  README.md                      problem, data access, results, how to reproduce
  src/fannie_default/
    layout.py                    column positions + dtypes for the fields used
    load.py                      chunked reading + deterministic loan sampling
    labels.py                    one row per loan: origination features + default_24m
    build.py                     CLI: raw quarterly files -> data/processed/loans.parquet
    vintage.py                   groupby default-rate tables
    features.py                  feature lists, preprocessor, out-of-time split
    models.py                    the three model factories
    evaluate.py                  metrics, cost threshold, permutation importance
  notebooks/
    01_vintage_analysis.ipynb
    02_model_comparison.ipynb
  reports/                       metrics.json + figures (committed; aggregates only)
  tests/
    conftest.py                  synthetic fixture writer
    test_load.py  test_labels.py  test_vintage.py  test_features.py
    test_models.py  test_evaluate.py
```

---

### Task 0: Data access (Matthew)

- [ ] **Step 1:** Register at capitalmarkets.fanniemae.com → Data Dynamics → Single-Family Loan Performance.
- [ ] **Step 2:** Download the twelve quarterly files `2015Q1`–`2017Q4` and the file-layout glossary (`crt-file-layout-and-glossary.xlsx`) into `~/Documents/GitHub/fannie-loan-default/data/raw/`.

Tasks 1–2 don't need the data; Task 3's real-data step does.

### Task 1: Scaffold and column layout

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `src/fannie_default/__init__.py`, `src/fannie_default/layout.py`, `tests/conftest.py`

**Interfaces:**
- Produces: `layout.FIELDS: dict[str, int]` (field name → 0-based column position), `layout.DATE_FIELDS = ("ACT_PERIOD", "ORIG_DATE", "FIRST_PAY")`, `layout.ORIGINATION_FIELDS: tuple[str, ...]`; and a fixture `raw_quarter_file(tmp_path, rows: list[dict]) -> Path` in `conftest.py` that writes rows in the real pipe-delimited, headerless format (positions from `FIELDS`; every other column blank).

- [ ] **Step 1:** `uv init --lib --python 3.12`, then `uv add pandas pyarrow scikit-learn lightgbm matplotlib` and `uv add --dev pytest jupyter`. Gitignore `data/`.
- [ ] **Step 2:** Write `FIELDS` for exactly these fields: `LOAN_ID, ACT_PERIOD, CHANNEL, ORIG_RATE, ORIG_UPB, ORIG_TERM, ORIG_DATE, FIRST_PAY, OLTV, OCLTV, NUM_BO, DTI, CSCORE_B, CSCORE_C, FIRST_FLAG, PURPOSE, PROP, NO_UNITS, OCC_STAT, STATE, MI_PCT, DLQ_STATUS, Zero_Bal_Code`. `ORIGINATION_FIELDS` is everything except `LOAN_ID, ACT_PERIOD, DLQ_STATUS, Zero_Bal_Code`.
- [ ] **Step 3: Verify the positions against the official glossary** once Task 0 is done. The positions in step 2 are from memory of Fannie's layout and must be checked field by field. Record the glossary's version or date in a comment at the top of `layout.py`.
- [ ] **Step 4:** `test_fixture_roundtrip`: a fixture row read back with `pd.read_csv(sep="|", header=None, dtype=str)` has `LOAN_ID` at column `FIELDS["LOAN_ID"]`. Run `uv run pytest -v` → PASS.
- [ ] **Step 5:** Commit `chore: scaffold project and column layout`.

### Task 2: Chunked loading with loan-level sampling

**Files:** Create `src/fannie_default/load.py`, `tests/test_load.py`

**Interfaces:**
- Consumes: `layout.FIELDS`, `layout.DATE_FIELDS`
- Produces: `in_sample(loan_id: str, rate: float) -> bool`; `read_quarter(path: Path, rate: float, chunksize: int = 500_000) -> pd.DataFrame`. The returned frame has named columns from `FIELDS`, date fields as `pd.Period[M]`, and numeric fields as floats (blank → NaN).

- [ ] **Step 1: Write failing tests**
  - `test_in_sample_deterministic`: same ID → same answer across calls; `rate=1.0` → always True; `rate=0.0` → always False.
  - `test_in_sample_rate`: over 20,000 synthetic IDs, the kept share at `rate=0.1` is within `0.09`–`0.11`.
  - `test_dates_keep_leading_zero`: `ACT_PERIOD="012016"` → `pd.Period("2016-01", "M")`.
  - `test_sampled_loan_keeps_all_rows`: a 30-row loan read with `chunksize=7` and `rate=1.0` returns all 30 rows.
- [ ] **Step 2:** Run `uv run pytest tests/test_load.py -v` → FAIL (module missing).
- [ ] **Step 3:** Implement. `in_sample` uses `zlib.crc32(loan_id.encode()) % 10_000 < rate * 10_000` (stable across runs, unlike `hash()`). `read_quarter` reads with `usecols=list(FIELDS.values())`, `dtype=str`, `chunksize`, filters each chunk with `in_sample`, concatenates, then converts types.
- [ ] **Step 4:** Run the tests → PASS.
- [ ] **Step 5:** Commit `feat: chunked quarterly loader with deterministic loan sampling`.

### Task 3: Loan-level table and default label

**Files:** Create `src/fannie_default/labels.py`, `src/fannie_default/build.py`, `tests/test_labels.py`

**Interfaces:**
- Consumes: `read_quarter`, `layout.ORIGINATION_FIELDS`
- Produces: `DEFAULT_ZB_CODES = {"02", "03", "09", "15"}`, `WINDOW_MONTHS = 24`; `build_loans(perf: pd.DataFrame) -> pd.DataFrame` → one row per `LOAN_ID` with the `ORIGINATION_FIELDS` (taken from the loan's earliest `ACT_PERIOD` row), `vintage: pd.Period[Q]` (from `ORIG_DATE`), and `default_24m: int`. CLI `uv run python -m fannie_default.build --raw data/raw --quarters 2015Q1 2017Q4 --rate 0.10 --out data/processed/loans.parquet`.

- [ ] **Step 1: Write failing tests** (synthetic per-loan monthly histories):
  - `test_90_day_delinquency_is_default`: `DLQ_STATUS` hits `3` in month 10 → `default_24m == 1`.
  - `test_late_delinquency_outside_window`: first hits `3` in month 30, with 36 months observed → `0`.
  - `test_foreclosure_code_is_default`: terminates in month 15 with `Zero_Bal_Code="09"` → `1`.
  - `test_prepaid_is_not_default`: terminates in month 8 with `"01"` → `0` and kept, not dropped as censored.
  - `test_unknown_status_is_missing`: statuses `"XX"`/blank in some months, the rest `0`, 24 months observed → `0`, no exception.
  - `test_censored_loan_dropped`: 12 months observed, no termination, no delinquency → not in the output.
  - `test_features_from_first_record`: `CSCORE_B` taken from the earliest `ACT_PERIOD` even if rows arrive out of order.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement `build_loans` with a `groupby("LOAN_ID")`. The window runs from `FIRST_PAY` to `FIRST_PAY + 23` months. Convert `DLQ_STATUS` with `pd.to_numeric(errors="coerce")`. `build.py` runs `read_quarter` for each quarter in the range, concatenates, calls `build_loans` once over all rows, writes parquet, and prints row count and default rate.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5 (needs Task 0):** Run the CLI on the real files. Expected: roughly 300k–500k loans and a default rate under 1%. A rate above 5% means a layout or label bug: stop and check the Task 1 positions.
- [ ] **Step 6:** Commit `feat: loan-level table with 24-month default label`.

### Task 4: Vintage analysis

**Files:** Create `src/fannie_default/vintage.py`, `tests/test_vintage.py`, `notebooks/01_vintage_analysis.ipynb`

**Interfaces:**
- Consumes: the `loans.parquet` schema from Task 3
- Produces: `default_rate_by(loans: pd.DataFrame, by: str | list[str]) -> pd.DataFrame` with columns `loans`, `defaults`, `default_rate`; `add_bands(loans) -> pd.DataFrame`, which adds `fico_band` (`<620, 620-679, 680-739, 740+`, plus `missing`) and `ltv_band` (`<=80, 80-90, 90-95, >95`).

- [ ] **Step 1:** Tests `test_rate_by_vintage` (hand-checked 3-vintage toy frame), `test_rate_by_two_keys` (vintage × fico_band gives a MultiIndex), `test_missing_fico_band`.
- [ ] **Step 2:** Run → FAIL; implement with `groupby(...).agg(...)`, `observed=True`; run → PASS.
- [ ] **Step 3:** Notebook: default rate by vintage (line chart), the vintage × FICO band pivot table and a heatmap of it, and a 2–3 sentence takeaway. Save figures to `reports/`.
- [ ] **Step 4:** Commit `feat: vintage default-rate analysis`.

### Task 5: Features and out-of-time split

**Files:** Create `src/fannie_default/features.py`, `tests/test_features.py`

**Interfaces:**
- Produces: `NUMERIC = ["ORIG_RATE", "ORIG_UPB", "ORIG_TERM", "OLTV", "OCLTV", "NUM_BO", "DTI", "CSCORE_B", "CSCORE_C", "MI_PCT"]`, `CATEGORICAL = ["CHANNEL", "FIRST_FLAG", "PURPOSE", "PROP", "NO_UNITS", "OCC_STAT", "STATE"]`; `out_of_time_split(loans, test_start: str = "2017Q1") -> tuple[X_train, X_test, y_train, y_test]`; `make_preprocessor(scale: bool) -> ColumnTransformer`.

- [ ] **Step 1:** Tests:
  - `test_split_is_by_vintage`: no test vintage earlier than `2017Q1`, no train vintage at or after it.
  - `test_no_performance_fields`: `X_train` columns ⊆ `NUMERIC + CATEGORICAL`.
  - `test_missing_scores_imputed`: a frame with blank `CSCORE_B` and `DTI` transforms with no NaN and gains missing-indicator columns.
  - `test_unseen_category`: a `STATE` value only in the test set doesn't raise.
- [ ] **Step 2:** Run → FAIL; implement: median `SimpleImputer(add_indicator=True)`, plus `StandardScaler` when `scale`; `OneHotEncoder(handle_unknown="ignore", min_frequency=0.005)`. Run → PASS.
- [ ] **Step 3:** Commit `feat: feature preprocessing and out-of-time split`.

### Task 6: Models (three learning gates)

**Files:** Create `src/fannie_default/models.py`, `tests/test_models.py`

**Interfaces:**
- Consumes: `make_preprocessor`
- Produces: `logistic() -> Pipeline`, `random_forest() -> Pipeline`, `boosting() -> Pipeline`. Each takes the raw `X` from `out_of_time_split` and supports `predict_proba`.

- [ ] **Step 1:** `logistic()`: `scale=True`, `LogisticRegression(class_weight="balanced", max_iter=2000)`. Test `test_model_fits_and_scores[logistic]` on a 2,000-row synthetic frame where default depends on `CSCORE_B`: probabilities in [0, 1] and AUC > 0.7. Run → PASS. Commit.
- [ ] **Gate A (Matthew):** `teach` session on random forests, bagging, OOB error and variable importance, with the quiz passed. Lesson log under `wiki/concepts/classes/stat-4630/sessions/`.
- [ ] **Step 2:** `random_forest()`: `scale=False`, `RandomForestClassifier(n_estimators=500, min_samples_leaf=50, class_weight="balanced_subsample", oob_score=True, n_jobs=-1, random_state=42)`. Same parametrized test. Commit.
- [ ] **Gate B (Matthew):** `teach` session on gradient boosting (learning rate, number of trees, early stopping; how it differs from random forests).
- [ ] **Step 3:** `boosting()`: `scale=False`, `LGBMClassifier(n_estimators=2000, learning_rate=0.03, num_leaves=31, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1)`. Early stopping on a 2016Q4 validation slice happens in the notebook, not the factory. Same test. Commit `feat: random forest and boosting models`.

### Task 7: Evaluation and comparison

**Files:** Create `src/fannie_default/evaluate.py`, `tests/test_evaluate.py`, `notebooks/02_model_comparison.ipynb`, `reports/metrics.json`

**Interfaces:**
- Produces: `score(y, p) -> dict` with keys `roc_auc`, `pr_auc`, `brier`, `base_rate`; `cost_threshold(y, p, cost_fn: float, cost_fp: float) -> tuple[float, float]` (threshold, total cost), minimizing `cost_fn·FN + cost_fp·FP` over the unique values of `p`; `importance(model, X, y, n_repeats=5) -> pd.Series` (permutation importance on ROC AUC, sorted descending).

- [ ] **Step 1:** Tests:
  - `test_cost_threshold_toy`: a hand-solved 6-point example gives the expected threshold.
  - `test_cost_threshold_extremes`: when FN cost is far higher than FP cost the threshold goes to the lowest score (flag everyone), and the reverse sends it above the highest.
  - `test_score_keys`.
- [ ] **Step 2:** Run → FAIL; implement; run → PASS.
- [ ] **Step 3:** Notebook: fit all three on train (LightGBM with early stopping on the 2016Q4 slice), score on the 2017 test set, and draw ROC and PR curves on one figure each. Then a calibration plot, cost-threshold results at `cost_fn/cost_fp = 10` and `25`, permutation importance for the best model, and RF OOB score vs. test AUC. Write `reports/metrics.json`.
- [ ] **Step 4:** Commit `feat: model comparison and cost-based threshold`.

### Task 8: Write-up and resume hand-off

**Files:** Create `README.md`; vault edits (follow the vault `CLAUDE.md`)

- [ ] **Step 1:** README: question, data access (no data shipped), label definition, why the test is out-of-time, a results table from `metrics.json`, the key figures, limitations (sampled, origination-only, pre-COVID window), and how to reproduce.
- [ ] **Step 2 (Matthew):** The explainability check. Explain without notes why PR AUC matters here, what the cost threshold means, and why the random forest's OOB score and its test AUC differ.
- [ ] **Step 3:** Draft one resume bullet with the real numbers. Edit the resume only via the backlog's "How to add a skill" procedure, after the bar is met. Tick the backlog rows (random forests, OOB/variable importance, gradient boosting, pandas groupby/datetime). Add a `build` log entry.
- [ ] **Step 4:** Commit `docs: README with results`.

---

## Decisions made by the plan

These are the plan author's calls, not Matthew's; change any of them before execution.

- **24-month window, 90+ DPD or a loss-type termination.** A common industry definition of "serious default" that keeps the label observable for all twelve vintages.
- **2015–2017 vintages.** The 24-month windows end before the March 2020 COVID forbearance wave, which would otherwise inflate delinquency for reasons the origination features can't see.
- **Out-of-time test (2017 vintages)** rather than a random split, because that's how a credit model is actually used. It's a good interview point.
- **10% loan sample**, hashed on `LOAN_ID`, to keep the build in memory on one machine.
- **pandas throughout** (no DuckDB), because pandas groupby and datetime are skills this build is meant to back.
- **LightGBM** over sklearn's `HistGradientBoostingClassifier` as the more recognizable industry tool.
