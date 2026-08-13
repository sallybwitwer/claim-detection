# Claim Detection

Binary sentence-level claim detection (`0` = not a claim, `1` = claim) with fine-tuned
encoders — `bert-base-uncased` and `answerdotai/ModernBERT-base` — exposed through a CLI
and a small HTTP API.

For the experiment itself — results, comparison against the source paper, and how to
interpret the numbers — see **[REPORT.md](REPORT.md)**.

## Setup

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Requires `transformers>=4.48` for ModernBERT support. Python 3.9+.

## Data

`data/ours/{train,test}.csv`, columns `text,label` — 10,397 training rows and 2,600 test
rows, near-balanced. Training data is cleaned at load time by `src.data.clean_train`,
which drops 74 rows (empty, duplicated, or leaking into test) and records what it removed
in each run's `manifest.json`. `test.csv` is never modified. See
[REPORT.md](REPORT.md#data) for the dataset's composition and why the cleaning matters.

## Fine-tuning

```bash
# Quick smoke test (~1 min) -- proves the whole path end to end
.venv/bin/python scripts/finetune.py --model bert-base-uncased \
    --max-train-samples 200 --epochs 1 --run-name smoke_bert

# 3-epoch runs
.venv/bin/python scripts/finetune.py --model bert-base-uncased --lr 5e-5 \
    --epochs 3 --batch-size 16
.venv/bin/python scripts/finetune.py --model answerdotai/ModernBERT-base --lr 5e-5 \
    --epochs 3 --batch-size 16

# 5-epoch runs (matching the paper)
.venv/bin/python scripts/finetune.py --model bert-base-uncased --lr 5e-5 \
    --epochs 5 --batch-size 16 --run-name bert-base-uncased_lr5e-05_e5_s42
.venv/bin/python scripts/finetune.py --model answerdotai/ModernBERT-base --lr 5e-5 \
    --epochs 5 --batch-size 16 --run-name answerdotai_ModernBERT-base_lr5e-05_e5_s42
```

Epochs and batch size are passed explicitly because the script's own defaults
(`--epochs 5 --batch-size 32`) do not match the recorded runs, which all used batch
size 16.

**Pass `--run-name` when changing the epoch count.** The default run name is
`{model}_lr{lr}_s{seed}` and does not encode epochs, so a 5-epoch run would otherwise
overwrite the 3-epoch results in place.

Useful flags: `--lr`, `--epochs`, `--batch-size`, `--max-length`, `--seed`, `--dev-size`,
`--data-dir`, `--output-root`, `--run-name`, `--tensorboard`.

Each run writes to `results/runs/{run_name}/`:

- `metrics.json` — dev and test metrics
- `manifest.json` — all arguments, device, library versions, and the cleaning report
- `checkpoint-*/` — model weights (gitignored; 1.2–1.7 GB each)

and per-example test predictions to `results/preds/{run_name}.csv`
(`text, label, pred, prob_1`).

To compare all runs at once:

```bash
.venv/bin/python -c "
import json, glob
for f in sorted(glob.glob('results/runs/*/metrics.json')):
    m = json.load(open(f)); t = m['test']
    print('%-45s macroF1 %.4f  P %.4f  R %.4f  F1pos %.4f' % (
        m['run'], t['macro_f1'], t['precision_pos'], t['recall_pos'], t['f1_pos']))
"
```

## Predicting on new text

```bash
.venv/bin/python scripts/predict.py --model MODERNBERT \
    --text "The unemployment rate fell to 3.4% in January 2023." \
           "I think we should all try to be kinder to one another."
```

```
claim  p(claim)=1.0000  The unemployment rate fell to 3.4% in January 2023.
not_claim  p(claim)=0.0000  I think we should all try to be kinder to one another.
```

Two arguments, both required. `--model` takes `BERT` or `MODERNBERT` in any casing;
`--text` takes any number of sentences. Sentences are truncated at 128 tokens, matching
training.

Each name maps to a fixed checkpoint in the `Model` enum in `enums.py` — edit those paths
to serve a different run:

| `--model` | Checkpoint | Test macro-F1 |
|---|---|---|
| `BERT` | `results/runs/bert-base-uncased_lr5e-05_e5_s42/checkpoint-2905` | 0.9114 |
| `MODERNBERT` | `results/runs/answerdotai_ModernBERT-base_lr5e-05_e5_s42/checkpoint-1743` | 0.9141 |

Both currently point at the 5-epoch runs. Note that the **3-epoch ModernBERT is the
stronger model** (0.9208) — point `Model.MODERNBERT` at
`answerdotai_ModernBERT-base_lr5e-05_s42/checkpoint-1743` to serve it.

**Checkpoints are not in this repository** — they are 1.2–1.7 GB each, well past GitHub's
100 MB per-file limit, and are gitignored. A fresh clone has no weights and the script
exits with the missing path; run the fine-tuning commands above first.

One caveat on the probabilities: `p(claim)` saturates at 0.0000 and 1.0000 because the
models are badly calibrated (see [REPORT.md](REPORT.md#rising-dev-loss)). Use it for the
decision and for ranking, but do not read it as a confidence level.

## Running the API

```bash
.venv/bin/uvicorn src.api:app --reload
```

Serving on `http://127.0.0.1:8000`; drop `--reload` if you don't want restarts on file
changes. **Run it from the repo root** — `src/api.py` imports `scripts.predict` and
`enums`, which resolve relative to the working directory, so starting it elsewhere fails
at import.

```bash
curl -X POST http://127.0.0.1:8000/detect-claims \
  -H 'Content-Type: application/json' \
  -d '{"model":"MODERNBERT","claims":["The unemployment rate fell to 3.4% in January 2023.","I think we should all be kinder."]}'
```

```json
{"predictions":[
  {"text":"The unemployment rate fell to 3.4% in January 2023.","label":"claim","prob_claim":1.0},
  {"text":"I think we should all be kinder.","label":"not_claim","prob_claim":7.6e-07}
]}
```

`model` accepts `BERT` or `MODERNBERT` in any casing; an unknown name returns 400 with
`{"detail":"unknown model 'RoBERTa'; choose from ['BERT', 'MODERNBERT']"}`. Malformed
requests return 422.

FastAPI generates interactive docs at **http://127.0.0.1:8000/docs**, which is an easier
way to poke at the endpoint than curl.

**The model is loaded from disk on every request** — ~570 MB for ModernBERT, so each call
takes a few seconds, nearly all of it loading rather than inference. Fine for trying the
API out, but it will not survive real traffic or concurrent requests. Caching each model
in memory at startup (a module-level cache, or FastAPI's `lifespan` handler) is the fix
when that matters.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

`tests/test_api.py` covers the `/detect-claims` endpoint: response shape, one prediction
per claim, claims passed through untouched, empty input, case-insensitive model names, a
400 for unknown models, and 422s for malformed requests. It runs in about 5 seconds
because `predict` is replaced with a fake — the real one loads a ~570 MB checkpoint that
is not in the repository.

One test does exercise the real model and is skipped by default:

```bash
RUN_INTEGRATION=1 .venv/bin/python -m pytest tests/ -q
```

It needs the BERT checkpoint present under `results/runs/`, and stays skipped if it is
absent, so the command is safe to run on a fresh clone.

## The code

```
enums.py              Model enum: checkpoint paths by model name
conftest.py           puts the repo root on sys.path for pytest
scripts/              entry points (fine-tuning, prediction)
src/                  library code (data, models, metrics, API)
tests/                API tests
data/ours/            train.csv, test.csv
results/runs/         per-run metrics.json, manifest.json, checkpoints
results/preds/        per-example test predictions
docs/                 session logs
```

### `scripts/finetune.py`

The fine-tuning CLI, and the only entry point that trains anything. Wraps the Hugging Face
`Trainer`: loads and cleans the data, carves a stratified dev split, tokenizes, trains, and
then evaluates the best checkpoint on test exactly once. Writes `metrics.json`,
`manifest.json` and a per-example predictions CSV per run. `--model` takes any Hugging Face
model id, so it is not limited to the two encoders reported here.

Two settings here are deliberate rather than incidental: mixed precision is disabled
(unreliable on MPS) and checkpoints are selected on macro-F1 rather than loss — see the
[report](REPORT.md#rising-dev-loss) for why that choice changes which model you end up
with.

### `scripts/predict.py`

Runs a fine-tuned checkpoint on new sentences. `predict(model, claims)` is the reusable
core — it takes a model name and a list of strings and returns one dict per sentence
(`text`, `label`, `prob_claim`) — and is what `src/api.py` calls. The rest of the file is
a thin argparse wrapper that prints those results. Model names are normalized to
upper case, so any casing works from either entry point.

### `src/data.py`

Everything that touches the CSVs. `load_splits()` reads train and test; `clean_train()`
drops empty, duplicated and test-leaking rows and returns a report of what it removed
(recorded in each run's manifest); `make_dev_split()` carves a stratified 10% dev set;
`to_dataset()` tokenizes into a `datasets.Dataset` without padding, since the collator
pads per batch instead. Shared by every model arm, so cleaning cannot drift between
experiments.

### `src/models.py`

Builds a model and tokenizer from a Hugging Face id. Mostly a thin wrapper, but it is
where the ModernBERT-on-Apple-silicon switches live: `attn_implementation="sdpa"` because
flash-attention is CUDA-only, and `reference_compile=False` because ModernBERT opts into
`torch.compile` by default, which MPS does not support. Also holds the label mapping
(`0 = not_claim`, `1 = claim`) and `model_slug()` for run naming.

### `src/metrics.py`

Scoring. `scores()` returns accuracy, macro-F1 and positive-class precision/recall/F1;
`compute_metrics()` is the `Trainer`-compatible wrapper around it, and macro-F1 from here
is what drives checkpoint selection. `predictions_frame()` builds the per-example CSV that
a later McNemar test will consume, and `softmax()` converts logits to probabilities for
both the CSV and the prediction paths.

### `src/api.py`

A FastAPI app with one endpoint, `POST /detect-claims`, taking `{"model": ..., "claims":
[...]}` and returning a prediction per claim. `resolve_model()` matches the requested name
against the `Model` enum case-insensitively and raises a 400 for anything unknown; the
prediction itself is delegated to `scripts.predict.predict`, so the API and the CLI cannot
diverge in behaviour.

## Apple silicon notes

These runs use the MPS backend in fp32. Mixed precision (`fp16`/`bf16`) is disabled in
`scripts/finetune.py` because it is unreliable on MPS, and these models are small enough
that fp32 costs little. `dataloader_num_workers=0` avoids a known hang with forked workers
on MPS. See `src/models.py` for the two ModernBERT-specific settings.
