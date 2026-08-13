# Fine-tuning BERT and ModernBERT for claim detection

An encoder comparison on `data/ours`: `bert-base-uncased` vs
`answerdotai/ModernBERT-base`, replicating and extending **Bell (2025), "Less Can be More:
An Empirical Evaluation of Small and Large Language Models for Sentence-level Claim
Detection," FEVER 2025** (`2025.fever-1.6.pdf` in this repo).

For how to run any of this, see [README.md](README.md).

---

## Data

`data/ours` is the composite dataset released with the paper. The match is exact:

| | Paper | `data/ours` |
|---|---|---|
| Total records | 12,997 | 12,997 |
| % positive | 47.83% | 47.83% |
| Test share | 20% | 20.00% |

so these runs are a replication on the identical train/test split, not merely a similar
task. It combines three human-curated sources:

| Source | Records | % positive |
|---|---|---|
| Claimbuster (Hassan et al., 2017) | 7,976 | 25.00 |
| PoliClaim Gold (Ni et al., 2024) | 1,953 | 59.09 |
| AVeriTeC (Schlichtkrull et al., 2023) | 3,068 | **100.00** |

Claimbuster and PoliClaim are US political speeches and debates; AVeriTeC is claims
published by fact-checking organizations. That AVeriTeC is entirely positive matters for
interpreting the results — see [How to read these numbers](#how-to-read-these-numbers).

| Split | Rows | Label 0 | Label 1 |
|---|---|---|---|
| train.csv | 10,397 | 5,394 | 5,003 |
| test.csv | 2,600 | 1,387 | 1,213 |

Texts are short — mean 18 words, 95th percentile 42, max 152 — so the 128-token cap
truncates nothing meaningful.

### Cleaning

The training split arrived with three defects, all removed by `src.data.clean_train` and
recorded in each run's `manifest.json`:

| Removed | Count |
|---|---|
| Empty / null `text` | 1 |
| Duplicate `text` | 56 |
| `text` also present in test.csv (leakage) | 17 |
| **Remaining** | **10,323** |

The 17 leaking rows matter most: left in, they inflate test scores by letting the model
memorize examples it is later graded on. `test.csv` is never modified. A stratified 10%
dev split (1,033 rows) is then carved out for checkpoint selection, leaving 9,290 rows for
training — so the test split is scored exactly once, at the end, and never informs any
decision.

## Method

Both models were fine-tuned with identical settings: learning rate 5e-5, batch size 16,
max sequence length 128, warmup ratio 0.1, weight decay 0.01, seed 42, fp32 on an Apple M1
Pro (16 GB) via the MPS backend. Checkpoints are selected by dev macro-F1.

**We ran 3 epochs first, then repeated both models at 5 epochs.** The 3-epoch runs came
first as a reasonable default; we then discovered the paper trains its encoders for
5 epochs ("sufficient for training loss to converge to close to 0") and reran to make the
comparison against it like-for-like. Both sets of numbers are reported below, because the
result was not what we expected: **5 epochs made both models slightly worse.**

## Results

Test set, n = 2,600. Single run per configuration.

| Model | Epochs | Accuracy | Macro-F1 | Precision (claim) | Recall (claim) | F1 (claim) |
|---|---|---|---|---|---|---|
| `bert-base-uncased` | 3 | 0.9135 | **0.9130** | 0.9117 | 0.9019 | 0.9068 |
| `bert-base-uncased` | 5 | 0.9119 | 0.9114 | 0.9177 | 0.8912 | 0.9042 |
| `answerdotai/ModernBERT-base` | 3 | **0.9212** | **0.9208** | 0.9131 | **0.9184** | **0.9157** |
| `answerdotai/ModernBERT-base` | 5 | 0.9146 | 0.9141 | 0.9153 | 0.9002 | 0.9077 |

The 3-epoch runs are the better ones. Going to 5 epochs cost BERT 0.2 macro-F1 points and
ModernBERT 0.7, and shifted both toward precision at recall's expense — BERT's recall fell
1.1 points, ModernBERT's 1.8, while both gained slightly on precision.

ModernBERT leads BERT at both epoch counts, but by a margin that shrinks from 0.78
macro-F1 points at 3 epochs to 0.27 at 5.

Training times at 3 epochs: 9.2 min (BERT), 13.3 min (ModernBERT).

## Training behaviour

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

### The 3- and 5-epoch runs are not nested

It is tempting to assume a 5-epoch run simply continues where a 3-epoch run stopped, so
that checkpoint selection could only ever pick a better model. That is wrong, and the
table above shows why: the 5-epoch run's epoch 3 scores 0.9176 for BERT, while the
3-epoch run's epoch 3 scores 0.9244.

The learning-rate schedule spans *total training steps*, not epochs — warmup over 174
steps then linear decay to zero across 1,743 at 3 epochs, versus warmup over 290 and decay
across 2,905 at 5. Every epoch therefore trains at a different effective learning rate, so
the two runs follow genuinely different trajectories. **Checkpoint selection protects you
within a run, not across runs with different schedules**, which is how more epochs
produced a worse final model here.

### Rising dev loss

Dev loss rose after epoch 1 in every run while dev F1 kept climbing. The models grow
overconfident on examples they already classify correctly: on the final ModernBERT
checkpoint, 205 wrong predictions (7.9% of test) account for **97.9% of total loss**, and
125 of them are held at >99% confidence in the wrong class.

| Dev loss | Dev F1 | Reading |
|---|---|---|
| ↑ | ↓ | Genuine overfitting — stop early |
| ↑ | ↑ | Confidence sharpening — keep training, select on F1 |

Ours is the second row, which is why checkpoints are selected on macro-F1 rather than
loss. Selecting on loss would have picked the epoch-1 checkpoint in both 3-epoch runs and
shipped a measurably worse classifier — BERT at 0.9176 dev F1 instead of 0.9244.

A side effect: `prob_1` in the prediction CSVs is badly calibrated. Harmless for
accuracy/F1/McNemar, which read only the argmax, but temperature scaling fit on dev would
be needed before using those probabilities for thresholding or abstention.

## Comparison with the paper

The paper's Precision/Recall/F1 are **positive-class**, not macro — verified
arithmetically, since the harmonic mean of its reported P and R reproduces its reported F1
to four decimals for both models. The paper trains for 5 epochs, so its runs compare
against ours at 5:

| Model | Metric | Paper | Ours (5 epochs) | Δ | Ours (3 epochs) | Δ |
|---|---|---|---|---|---|---|
| **BERT** | Precision | 0.918 | 0.9177 | −0.000 | 0.9117 | −0.006 |
| | Recall | 0.904 | 0.8912 | −0.013 | 0.9019 | −0.002 |
| | **F1** | **0.911** | **0.9042** | **−0.007** | **0.9068** | **−0.004** |
| | Accuracy | 0.917 | 0.9119 | −0.005 | 0.9135 | −0.004 |
| **ModernBERT** | Precision | 0.907 | 0.9153 | +0.008 | 0.9131 | +0.006 |
| | Recall | 0.902 | 0.9002 | −0.002 | 0.9184 | +0.016 |
| | **F1** | **0.904** | **0.9077** | **+0.004** | **0.9157** | **+0.012** |
| | Accuracy | 0.911 | 0.9146 | +0.004 | 0.9212 | +0.010 |

**At the paper's own epoch count, both models replicate within 0.013 on every metric.**
Matching the setup tightened ModernBERT considerably — +0.004 F1 at 5 epochs versus +0.012
at 3 — while BERT drifted marginally further but stayed within 0.007.

BERT's precision is the closest agreement of all: 0.9177 against the paper's 0.918. Its
recall is the widest gap at −0.013, and that single metric accounts for essentially all of
BERT's F1 shortfall.

Remaining differences in setup, which plausibly account for the residual gap:

1. **Training set size.** The paper appears to train on the full 80% (10,397 rows) with no
   dev split and no checkpoint selection; we hold out 10% for dev and train on 9,290.
2. **Our cleaning.** We drop 74 rows the paper keeps — including 17 whose text also
   appears in the test split — which makes our numbers slightly more conservative.
3. **Hardware and library versions.** The paper used an NVIDIA GTX 4060 Ti; these runs use
   Apple MPS in fp32.

### Where we disagree with the paper

The paper ranks **BERT** first (F1 0.911 vs ModernBERT's 0.904). We get the **reverse**
ordering at both epoch counts.

But the margin is small on both sides — 0.7 F1 points in the paper's direction, 0.27
macro-F1 points in ours at the matched setting — and neither experiment reports error
bars. The paper itself notes the two models share 94% of their positive predictions and
are "likely learning the same underlying semantic structures."

The defensible reading is that **the two models perform equivalently on this dataset**,
and that both experiments are ordering noise.

## How to read these numbers

Three caveats matter for any writeup:

1. **The BERT-vs-ModernBERT ordering is not established.** Single runs, no error bars, and
   a gap that shrinks by two-thirds when the epoch count changes.
2. **Epoch count moves results by more than the model choice does.** Going 3 → 5 epochs
   changed each model's macro-F1 by 0.2–0.7 points, comparable to or larger than the gap
   between the models. Any claim about which model is better has to hold the schedule
   fixed and vary the seed instead.
3. **The task may be easier than it looks.** A smoke run on 200 training examples for
   1 epoch already reached 0.857 test macro-F1; the full 9,290 examples reach only 0.913.
   Buying 5.6 points with 46× the data is a flat learning curve, characteristic of a task
   solvable from surface features — register, topic, source style — rather than
   claim-ness. The dataset composition supplies a mechanism: AVeriTeC contributes 3,068
   rows that are 100% positive, so nearly a quarter of the data has a label perfectly
   predictable from provenance, and its fact-check register differs sharply from
   Claimbuster's debate transcripts. The paper reaches a compatible conclusion from the
   other direction, finding that BERT-based models "transfer poorly … often over-predicting
   the positive outcome" on out-of-domain CheckThat tweets.

## Open items

- **Repeat seeds (43, 44)** — the highest-value next step, since the BERT-vs-ModernBERT
  ordering rests entirely on single runs and the gap is smaller than the epoch-count
  effect.
- **McNemar significance test** between the two encoders. The per-example predictions in
  `results/preds/` are the required input, and `scipy` is already pinned for it.
- **Learning curve** at 200 / 1,000 / 5,000 examples, to quantify caveat 3 above.
- **CheckThat out-of-domain evaluation** — the paper's most interesting result;
  `data/CheckThat/` is currently empty.
- **LLM arm:** prompt templates, LoRA fine-tuning on Llama-3.2-1B via `peft`, and the API
  adapter, to complete the comparison the paper makes.
