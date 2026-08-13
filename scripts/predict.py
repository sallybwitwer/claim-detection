"""Run a fine-tuned claim-detection model on new text.

    python scripts/predict.py --model ModernBERT \
        --text "The unemployment rate fell to 3.4% in January." \
               "I think we should all be kinder to one another."
"""

import argparse
import os
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.metrics import softmax  # noqa: E402

MAX_LENGTH = 128

CHECKPOINTS = {
    "BERT": "results/runs/bert-base-uncased_lr5e-05_e5_s42/checkpoint-2905",
    "ModernBERT": "results/runs/answerdotai_ModernBERT-base_lr5e-05_e5_s42/checkpoint-1743",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=sorted(CHECKPOINTS), help="which model")
    p.add_argument("--text", required=True, nargs="+", help="one or more sentences")
    return p.parse_args(argv)


def resolve_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@torch.no_grad()
def main(argv=None) -> int:
    args = parse_args(argv)

    ckpt = os.path.join(REPO_ROOT, CHECKPOINTS[args.model])
    if not os.path.isdir(ckpt):
        raise SystemExit(f"checkpoint not found: {ckpt}\nTrain the model first, or fix CHECKPOINTS.")

    # ModernBERT's torch.compile path is unsupported on MPS; sdpa replaces
    # flash-attention, which is CUDA-only (see src/models.py).
    kwargs = {"attn_implementation": "sdpa", "reference_compile": False}
    model = AutoModelForSequenceClassification.from_pretrained(
        ckpt, **(kwargs if args.model == "ModernBERT" else {})
    )
    tokenizer = AutoTokenizer.from_pretrained(ckpt)

    device = resolve_device()
    model.eval().to(device)

    enc = tokenizer(
        args.text, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt"
    ).to(device)
    probs = softmax(model(**enc).logits.float().cpu().numpy())

    for text, p in zip(args.text, probs):
        prediction = model.config.id2label[int(p.argmax())]
        print(f"{prediction}  p(claim)={p[1]:.4f}  {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
