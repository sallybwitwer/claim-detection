"""Run a fine-tuned claim-detection model on new text.

    # ad-hoc sentences
    python scripts/predict.py --run results/runs/answerdotai_ModernBERT-base_lr5e-05_s42 \
        --text "The unemployment rate fell to 3.4% in January."

    # a whole CSV
    python scripts/predict.py --run results/runs/bert-base-uncased_lr5e-05_s42 \
        --input-csv data/ours/test.csv --output-csv /tmp/preds.csv

Point --run at a run directory and the best checkpoint inside it is used, or pass
--checkpoint to load a specific one.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

from src.metrics import scores, softmax  # noqa: E402
from src.models import is_modernbert  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", help="run directory under results/runs/")
    src.add_argument("--checkpoint", help="a specific checkpoint-* directory")

    p.add_argument("--text", action="append", default=[], help="repeatable")
    p.add_argument("--input-csv", help="CSV to score")
    p.add_argument("--text-column", default="text")
    p.add_argument("--label-column", default="label", help="scored if present")
    p.add_argument("--output-csv", help="where to write predictions")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=128)
    return p.parse_args(argv)


def resolve_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_checkpoint(args) -> str:
    """Best checkpoint in a run dir, or the explicitly given one."""
    if args.checkpoint:
        return args.checkpoint

    candidates = sorted(glob.glob(os.path.join(args.run, "checkpoint-*")))
    if not candidates:
        raise SystemExit(f"no checkpoint-* directory found in {args.run}")
    if len(candidates) == 1:
        return candidates[0]

    # More than one survived rotation: trainer_state records which was best.
    for c in candidates:
        state_path = os.path.join(c, "trainer_state.json")
        if not os.path.exists(state_path):
            continue
        best = json.load(open(state_path)).get("best_model_checkpoint")
        if best and os.path.isdir(best):
            return best
    return candidates[-1]


def base_model_id(ckpt: str, run: str) -> str:
    """The HF id this checkpoint came from, for the ModernBERT/MPS switches."""
    manifest = os.path.join(run or os.path.dirname(ckpt), "manifest.json")
    if os.path.exists(manifest):
        return json.load(open(manifest))["args"]["model"]
    return ckpt


@torch.no_grad()
def predict(texts, model, tokenizer, device, batch_size=64, max_length=128):
    model.eval().to(device)
    out = []
    for i in range(0, len(texts), batch_size):
        batch = [str(t) for t in texts[i : i + batch_size]]
        enc = tokenizer(
            batch, truncation=True, max_length=max_length, padding=True, return_tensors="pt"
        ).to(device)
        out.append(model(**enc).logits.float().cpu().numpy())
    return np.concatenate(out) if out else np.empty((0, 2))


def main(argv=None) -> int:
    args = parse_args(argv)

    if not args.text and not args.input_csv:
        raise SystemExit("give --text (repeatable) or --input-csv")

    ckpt = resolve_checkpoint(args)
    # Checkpoints carry their own config and tokenizer, so they load standalone --
    # only the ModernBERT/MPS switches need re-applying (see src/models.py).
    kwargs = {}
    if is_modernbert(base_model_id(ckpt, args.run)):
        kwargs = {"attn_implementation": "sdpa", "reference_compile": False}
    model = AutoModelForSequenceClassification.from_pretrained(ckpt, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(ckpt)

    device = resolve_device()
    print(f"[model] {ckpt}  (device={device})", file=sys.stderr)

    if args.text:
        logits = predict(args.text, model, tokenizer, device, args.batch_size, args.max_length)
        probs = softmax(logits)
        for text, p in zip(args.text, probs):
            pred = int(p.argmax())
            print(f"{model.config.id2label[pred]:>10}  p(claim)={p[1]:.4f}  {text}")

    if args.input_csv:
        df = pd.read_csv(args.input_csv)
        logits = predict(
            df[args.text_column].tolist(), model, tokenizer, device, args.batch_size, args.max_length
        )
        probs = softmax(logits)
        df["pred"] = logits.argmax(axis=-1)
        df["prob_1"] = probs[:, 1]

        if args.label_column in df.columns:
            print(json.dumps(scores(df[args.label_column].to_numpy(), df["pred"].to_numpy()), indent=2))
        else:
            print(f"{int(df.pred.sum())} of {len(df)} predicted as claims")

        if args.output_csv:
            df.to_csv(args.output_csv, index=False)
            print(f"[out] {args.output_csv}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
