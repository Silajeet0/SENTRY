# Laya affiliation benchmark

The previously empty `benchmark_data.json` now contains 121 synthetic affiliation
cases: 40 positive, 40 negative and 41 ambiguous, plus 12 paper-level replay
fixtures. Every case has an ID, rationale, challenge category and fixed split
(24 validation / 97 test). Labels are editorial and need human review; this is
a diagnostic challenge set, not a representative scholarly sample or an
independent gold-standard dataset. Similar institutional names occur across
splits, so this split does not measure generalization to unseen institutions.

Positive means at least one **current institutional affiliation located in India**.
Negative requires identified foreign affiliations. Ambiguous means insufficient
evidence. Email, nationality, former jobs and a research subject mentioning India
do not establish location. Overseas campuses are classified by campus location.

## Run

```bash
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements-laya.txt
.venv/bin/python benchmark/test_laya.py --backend both --model convaiinnovations/laya

# Reproduce the checkpoint used in the included report:
.venv/bin/python benchmark/test_laya.py --backend both --model convaiinnovations/laya \
  --revision 7b928d828b7b0e022f929d9bd2e44165aa270148 \
  --output benchmark/results/reproduction.json

# No model or download needed:
.venv/bin/python benchmark/test_laya.py --backend rules --output benchmark/results/rules_only.json
```

The script uses the [official Laya SDK](https://github.com/NandhaKishorM/laya) and
the requested [convaiinnovations/laya checkpoint](https://huggingface.co/convaiinnovations/laya),
with one typed choice question and three fixed criteria. It does not substitute
an LLM or silently fall back to rules. Missing dependencies, download errors and
invalid model responses produce a failed report and nonzero exit. The model is
loaded once; inference uses CPU with two threads by default. `--device`,
`--threads`, `--model`, `--revision`, `--dataset`, `--split` and `--output` are
configurable. Model files are cached under ignored `data/model_cache`.

The report records the dataset SHA-256, complete question/criteria, model revision,
library versions, per-case distributions and timings, confusion matrices,
per-class precision/recall/F1, macro-F1, Brier score and ECE. Brier is the sum
across three classes averaged over cases; ECE uses ten equal-width bins.
Probabilities are normalized only to compensate for SDK rounding. Reported
latency excludes model loading but includes the first inference call.
No confidence threshold was tuned on the test set.

## Independent benchmark comparison

`benchmark/benchmark_data.json` is kept separate from the fine-tuning corpus. Its
three-class labels are converted by
[`parse_benchmark_data.py`](parse_benchmark_data.py) into the exact binary Laya
format (`state`, `questions`, `expected`). `positive` projects to `INDIAN`, while
`negative` and `ambiguous` project to `NOT_INDIAN`. The original source label and
category remain in the comparison report for analysis.

Run both systems on all 121 independent cases with:

```bash
.venv/bin/python benchmark/test_independent_laya_vs_rules.py
```

The command writes the parsed input to
`benchmark/results/benchmark_data_laya.jsonl` and the comparison to
`benchmark/results/independent_laya_vs_rules_report.json`. It also saves overall,
per-class, confusion-matrix and category-accuracy PNG plots in
`benchmark/results/`. Use
`--split test` or `--split validation` to restrict the source split, and
`--max-cases N` for a short smoke run.

## Measured result

`results/laya_report.json` records an actual CPU run on 2026-10-24:

| Metric | Existing rules | Laya |
|---|---:|---:|
| All 121 cases: accuracy | 62.81% | 33.88% |
| All 121 cases: macro-F1 | 0.610 | 0.270 |
| Test split, 97 cases: accuracy | 62.89% | 34.02% |
| Positive recall | 95.00% | 42.50% |
| Ambiguous recall | 31.71% | 0.00% |
| False inclusions / false exclusions | 9 / 2 | 31 / 23 |
| Paper-level three-class accuracy, 12 fixtures | 58.33% | 25.00% |

Laya predicted 48 positives, 73 negatives and **zero ambiguous** cases. Its mean
CPU latency was approximately 2.27 seconds per case, excluding loading. This
historical report evaluates the original three-class base-checkpoint experiment;
it is not the production checkpoint result. The current production pipeline uses
`laya_finetuning/laya_finetuned_SENTRY` and the binary contract.

During loading, the SDK warned that an unrelated 11+-option temperature entry
was clamped to its supported range. This benchmark uses three options. Confidence
still has no domain-specific calibration guarantee; the measured Brier/ECE are
diagnostics, not proof of calibration.

The paper replay groups structured author-affiliation decisions and applies the
pipeline's positive-only inclusion policy. It does **not** rerun web acquisition,
author extraction or form filling and is not a full end-to-end pipeline accuracy
measurement. The separate
`test_rag.py` script tests retrieval integration on synthetic paper fixtures.

For the Laya fine-tuning corpus, see
[`laya_finetuning/README.md`](laya_finetuning/README.md). It contains 1,200
binary JSONL examples (400 `INDIAN`, 800 `NOT_INDIAN`) in Laya's
`{state, questions, expected}` format, plus reproducible
train/validation/test splits.

The following comparison scripts are retained as historical diagnostics for the
former deterministic classifier. They are not imported by the production
pipeline and can be removed together with `evaluation/india_rules.py` when that
historical comparison is no longer needed. To reproduce the old comparison on
the untouched test split, run:

```bash
.venv/bin/python benchmark/test_laya_vs_rules.py
```

The default inputs are `laya_finetuning/finetuning_data/test.jsonl` and
`laya_finetuning/laya_finetuned_SENTRY`. The script writes
`benchmark/results/laya_vs_rules_test_report.json`. It maps the
rule classifier's `positive` result to `INDIAN` and both `negative` and
`ambiguous` results to `NOT_INDIAN`, while retaining the raw rule label in the
report.

For the Laya fine-tuning corpus, see
[`laya_finetuning/README.md`](laya_finetuning/README.md). It contains 1,200
binary JSONL examples (400 `INDIAN`, 800 `NOT_INDIAN`) in Laya's
`{state, questions, expected}` format, plus reproducible
train/validation/test splits.
