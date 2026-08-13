# Claim Detection

Binary claim detection (`0` = not a claim, `1` = claim) on `data/ours`, comparing
fine-tuned encoder models. This repo currently covers the **encoder arm**:
`bert-base-uncased` vs `answerdotai/ModernBERT-base`.

`data/ours` is the composite dataset released with **Bell (2025), "Less Can be More: An
Empirical Evaluation of Small and Large Language Models for Sentence-level Claim
Detection," FEVER 2025** (`2025.fever-1.6.pdf` in this repo), so these runs are a
replication on the identical train/test split. See
[Comparison with the paper](#comparison-with-the-paper).

## Results

Test set, n = 2,600. Single run per configuration, seed 42, learning rate 5e-5,
batch size 16, max sequence length 128. Checkpoint selected by dev macro-F1; the test
split was scored once, at the end. Apple M1 Pro (16 GB), MPS backend, fp32.

| Model | Epochs | Accuracy | Macro-F1 | Precision (claim) | Recall (claim) | F1 (claim) |
|---|---|---|---|---|---|---|
| `bert-base-uncased` | 3 | 0.9135 | **0.9130** | 0.9117 | 0.9019 | 0.9068 |
| `bert-base-uncased` | 5 | 0.9119 | 0.9114 | 0.9177 | 0.8912 | 0.9042 |
| `answerdotai/ModernBERT-base` | 3 | **0.9212** | **0.9208** | 0.9131 | **0.9184** | **0.9157** |
| `answerdotai/ModernBERT-base` | 5 | 0.9146 | 0.9141 | 0.9153 | 0.9002 | 0.9077 |

The 3-epoch runs are the better ones; 5 epochs cost both models roughly 0.2–0.7 macro-F1
points and shifted both toward precision at recall's expense. The 5-epoch runs exist to
match the paper's setup, not because they perform better.

ModernBERT leads BERT at both epoch counts, but by a margin that shrinks from 0.78
macro-F1 points at 3 epochs to 0.27 at 5 — see
[How to read these numbers](#how-to-read-these-numbers).

Training times at 3 epochs: 9.2 min (BERT) and 13.3 min (ModernBERT).

### Training behaviour

Learning rate 5e-5 was stable throughout — no divergence, train loss falling
monotonically. Dev macro-F1 by epoch:

| Epoch | BERT (3ep run) | BERT (5ep run) | ModernBERT (3ep run) | ModernBERT (5ep run) |
|---|---|---|---|---|
| 1 | 0.9176 | 0.9098 | 0.9116 | 0.9102 |
| 2 | 0.9196 | 0.8984 | 0.9157 | 0.9128 |
| 3 | **0.9244** | 0.9176 | **0.9254** | **0.9224** |
| 4 | — | 0.9148 | — | below 0.9224 |
| 5 | — | **0.9185** | — | below 0.9224 |

(ModernBERT's epoch 4–5 dev scores are not recoverable from disk — checkpoint rotation
kept only the best checkpoint — but both were below epoch 3, since epoch 3 was selected.)

**The 3-epoch and 5-epoch runs are not nested.** The learning-rate schedule spans total
training steps: warmup over 174 steps then decay to zero across 1,743 at 3 epochs, versus
warmup over 290 and decay across 2,905 at 5. Every epoch therefore trains at a different
effective learning rate, and the 5-epoch run's epoch 3 is not the 3-epoch run's epoch 3
(0.9176 vs 0.9244 for BERT). Checkpoint selection protects you *within* a run, not across
runs with different schedules — which is why more epochs can and here did produce a worse
final model.

Dev *loss* rose after epoch 1 in every run while dev F1 continued to climb. The models
grow overconfident on examples they already classify correctly: on the final ModernBERT
checkpoint, 205 wrong predictions (7.9% of test) account for 97.9% of total loss, and 125
of them are held with >99% confidence in the wrong class. This is why checkpoints are
selected on macro-F1 rather than loss — selecting on loss would have picked the epoch-1
checkpoint in both 3-epoch runs and shipped a measurably worse classifier.

A side effect: `prob_1` in the preds CSVs is badly calibrated. Harmless for
accuracy/F1/McNemar, which read only the argmax, but temperature scaling fit on dev would
be needed before using those probabilities for thresholding or abstention.

## Comparison with the paper

`data/ours` matches the paper's released dataset exactly:

| | Paper | `data/ours` |
|---|---|---|
| Total records | 12,997 | 12,997 |
| % positive | 47.83% | 47.83% |
| Test share | 20% | 20.00% |

It is a composite of Claimbuster (7,976 sentences, 25% positive), PoliClaim Gold
(1,953, 59.1%) and AVeriTeC (3,068, **100% positive**), split 80/20 and frozen.

The paper's Precision/Recall/F1 are **positive-class**, not macro — verified
arithmetically, since the harmonic mean of its reported P and R reproduces its reported F1
to four decimals for both models. The paper trains for 5 epochs, so its runs compare
against ours at 5:

| Model | Metric | Paper | Ours (5 epochs) | Δ |
|---|---|---|---|---|
| **BERT** | Accuracy | 0.917 | 0.9119 | −0.005 |
| | Precision | 0.918 | 0.9177 | −0.000 |
| | Recall | 0.904 | 0.8912 | −0.013 |
| | **F1** | **0.911** | **0.9042** | **−0.007** |
| **ModernBERT** | Accuracy | 0.911 | 0.9146 | +0.004 |
| | Precision | 0.907 | 0.9153 | +0.008 |
| | Recall | 0.902 | 0.9002 | −0.002 |
| | **F1** | **0.904** | **0.9077** | **+0.004** |

**At the paper's own epoch count, both models replicate within 0.013 on every metric.**

Remaining differences in setup, which plausibly account for the residual gap:

1. **Training set size.** The paper appears to train on the full 80% (10,397 rows) with no
   dev split and no checkpoint selection; we hold out 10% for dev and train on 9,290.
2. **Our cleaning.** We drop 74 rows the paper keeps — including 17 whose text also
   appears in the test split — which makes our numbers slightly more conservative.
3. **Hardware and library versions.** The paper used an NVIDIA GTX 4060 Ti; these runs use
   Apple MPS in fp32.

### Where we disagree with the paper

The paper ranks **BERT** first (F1 0.911 vs ModernBERT's 0.904). We get the **reverse**
ordering at both epoch counts. But the gap is small on both sides — 0.7 points in the
paper's direction, 0.27 in ours at the matched setting — and neither experiment reports
error bars. The paper itself notes the two models share 94% of their positive predictions
and are "likely learning the same underlying semantic structures."

The defensible reading is that **the two models perform equivalently on this dataset**,
and both experiments are ordering noise. Repeat seeds and a McNemar test would settle it.

## How to read these numbers

Three caveats matter for any writeup:

1. **The BERT-vs-ModernBERT ordering is not established.** Single runs, no error bars, and
   a gap that halves when the epoch count changes. See above.
2. **Epoch count moves results by more than the model choice does.** Going 3 → 5 epochs
   changed each model's macro-F1 by 0.2–0.7 points, comparable to or larger than the gap
   between the models. Any claim about which model is better needs to hold the schedule
   fixed and vary the seed.
3. **The task may be easier than it looks.** A smoke run on 200 training examples for
   1 epoch already reached 0.857 test macro-F1; the full 9,290 examples reach only 0.913.
   Buying 5.6 points with 46× the data is a flat learning curve, characteristic of a task
   solvable from surface features (register, topic, source style) rather than claim-ness.
   Table 1 of the paper supplies a mechanism: the AVeriTeC source contributes 3,068 rows
   that are **100% positive**, so nearly a quarter of the dataset has a label perfectly
   predictable from provenance. The paper reaches a compatible conclusion from the other
   direction, finding that BERT-based models "transfer poorly … often over-predicting the
   positive outcome" on out-of-domain CheckThat tweets.

## Data

`data/ours/{train,test}.csv`, columns `text,label`.

| Split | Rows | Label 0 | Label 1 |
|---|---|---|---|
| train.csv | 10,397 | 5,394 | 5,003 |
| test.csv | 2,600 | 1,387 | 1,213 |

Texts are short — mean 18 words, 95th percentile 42, max 152 — so `max_length=128`
truncates nothing meaningful.

Training data is cleaned at load time by `src.data.clean_train`, which removes
74 rows and records exactly what it dropped in each run's `manifest.json`:

| Removed | Count |
|---|---|
| Empty / null `text` | 1 |
| Duplicate `text` | 56 |
| `text` also present in test.csv (leakage) | 17 |
| **Remaining** | **10,323** |

`test.csv` is never modified. A stratified 10% dev split (1,033 rows) is carved from the
cleaned training data, leaving 9,290 rows for training.

## Setup

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Requires `transformers>=4.48` for ModernBERT support.

## Usage

```bash
# Quick smoke test (~1 min) -- proves the whole path end to end
.venv/bin/python scripts/finetune.py --model bert-base-uncased \
    --max-train-samples 200 --epochs 1 --run-name smoke_bert

# 3-epoch runs
.venv/bin/python scripts/finetune.py --model bert-base-uncased --lr 5e-5
.venv/bin/python scripts/finetune.py --model answerdotai/ModernBERT-base --lr 5e-5

# 5-epoch runs (matching the paper)
.venv/bin/python scripts/finetune.py --model bert-base-uncased --lr 5e-5 \
    --epochs 5 --run-name bert-base-uncased_lr5e-05_e5_s42
.venv/bin/python scripts/finetune.py --model answerdotai/ModernBERT-base --lr 5e-5 \
    --epochs 5 --run-name answerdotai_ModernBERT-base_lr5e-05_e5_s42
```

**Pass `--run-name` when changing the epoch count.** The default run name is
`{model}_lr{lr}_s{seed}` and does not encode epochs, so a 5-epoch run would otherwise
overwrite the 3-epoch results in place.

Useful flags: `--lr`, `--epochs`, `--batch-size`, `--max-length`, `--seed`,
`--dev-size`, `--data-dir`, `--output-root`, `--run-name`, `--tensorboard`.

Each run writes to `results/runs/{run_name}/`:

- `metrics.json` — dev and test metrics
- `manifest.json` — all arguments, device, library versions, and the cleaning report
- `checkpoint-*/` — model weights (gitignored)

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

## Repo layout

```
src/data.py       load_splits, clean_train, make_dev_split, to_dataset
src/metrics.py    scores, compute_metrics, predictions_frame
src/models.py     build_model_and_tokenizer
scripts/finetune.py   fine-tuning CLI over the above, using the HF Trainer
scripts/predict.py    run a saved checkpoint on new text or a CSV
docs/             session logs
```

### Apple silicon notes

`src/models.py` applies two settings for ModernBERT on MPS: `attn_implementation="sdpa"`
(flash-attention is CUDA-only) and `reference_compile=False` (ModernBERT opts into
`torch.compile` by default, which MPS does not support). Mixed precision is disabled in
`scripts/finetune.py` — it is unreliable on MPS, and these models are small enough that
fp32 costs little.

## Not yet implemented

- **Repeat seeds (43, 44)** — the highest-value next step, since the BERT-vs-ModernBERT
  ordering currently rests on single runs
- **McNemar significance test** between the two encoders (inputs are ready in
  `results/preds/`, `scipy` already pinned)
- **Learning curve** at 200 / 1,000 / 5,000 examples, for caveat 3 above
- **CheckThat out-of-domain evaluation** — the paper's most interesting result;
  `data/CheckThat/` is empty
- **LLM arm:** prompt templates, LoRA fine-tuning on Llama-3.2-1B via `peft`, and the API
  adapter
