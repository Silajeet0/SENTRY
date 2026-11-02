# Laya institution-geography fine-tuning data

This directory contains a reproducible, synthetic supervised corpus for adapting
Laya to SENTRY's binary affiliation decision:

- `INDIAN`: sufficient evidence that at least one current institutional
  affiliation is located in India;
- `NOT_INDIAN`: the affiliation is outside India or there is not enough reliable
  evidence to establish a current Indian affiliation.

There are **1,200 examples**: 400 `INDIAN` followed by 800 `NOT_INDIAN` rows. The
seven requested positive categories, seven foreign-negative categories and eight
former ambiguous categories are all represented. The files contain 960 training,
120 validation and 120 test rows, with 320/640, 40/80 and 40/80
`INDIAN`/`NOT_INDIAN` examples respectively. `all.jsonl` is the complete corpus.

The source templates originally carried three labels. Every former `ambiguous`
row is now a final `NOT_INDIAN` row because it does not establish a current Indian
affiliation.

Each JSONL row is directly compatible with Laya's native training/evaluation
reader:

```json
{"state":"IIT Bombay, Mumbai","questions":{"affiliation_geography":{"type":"choice","instructions":"Does this affiliation provide sufficient evidence that the current institution or campus is located in India?","criteria":{"INDIAN":"Sufficient evidence that the current institution or campus is in India.","NOT_INDIAN":"No sufficient evidence that the current institution or campus is in India."}}},"expected":{"affiliation_geography":"INDIAN"}}
```

Each row has one `expected` target field. Exact counts, hashes and policy are in
[manifest.json](manifest.json).

The generator is [generate_laya_finetuning.py](../generate_laya_finetuning.py).
Re-running it with the same code and seed (`20261029`) reproduces the same rows
and split assignments.

## Validate and train

```bash
.venv/bin/python benchmark/generate_laya_finetuning.py
.venv/bin/python - <<'PY'
from laya.evals import Dataset
for name in ("train", "validation", "test"):
    data = Dataset.from_jsonl(f"benchmark/laya_finetuning/{name}.jsonl")
    print(name, len(data))
PY

.venv/bin/laya-train \
  --data benchmark/laya_finetuning/train.jsonl \
  --eval benchmark/laya_finetuning/validation.jsonl \
  --base convaiinnovations/laya \
  --out data/models/laya-india-affiliation \
  --epochs 4 --micro-batch 8 --grad-accum 8 --loss rlcd --seed 20261029
```

On an offline machine, point `--base` at a downloaded local Laya checkpoint.
The command needs the Laya training dependencies and a writable output path.
Keep the test split untouched while choosing epochs, learning rates, or
abstention thresholds. Evaluate the resulting checkpoint with
`benchmark/test_laya.py --model data/models/laya-india-affiliation --split test`.

## Labeling cautions

This is generated training data, not an externally verified gold-standard
corpus. It uses manually written institution families and controlled variations.
The same families can appear across splits, so the test score measures coverage
of these policy patterns rather than generalization to unseen institutions.
The 1:2 `INDIAN`/`NOT_INDIAN` ratio is a valid choice when unresolved evidence is
operationally treated as negative; it is intentionally not a 50/50 class-balanced
corpus. Report per-class recall, macro-F1, balanced accuracy and the confusion
matrix in addition to accuracy. If deployment prevalence differs materially,
consider class weights or calibration during model selection rather than judging
the model by accuracy alone.
Review labels before training. In particular:

- A current Indian affiliation makes a multiple-affiliation example `INDIAN`;
  a foreign coauthor does not cancel it.
- A former Indian job does not make a current foreign affiliation `INDIAN`.
- A bare acronym, email address, city alone, India as a research topic, or any
  other unresolved evidence is `NOT_INDIAN` under the binary policy.
- An overseas campus is classified by its current campus location.
- `IIT`, `NIT`, `ISI`, `VIT` and similar acronyms are `NOT_INDIAN` without a
  disambiguating campus or institution.

For a serious model release, add reviewed examples from real affiliation strings,
hold out institution families by name, and record annotator agreement. Do not
use these synthetic labels alone to make employment, nationality or eligibility
decisions.
