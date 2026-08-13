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

from enums import Checkpoint  # noqa: E402
from src.metrics import softmax  # noqa: E402

MAX_LENGTH = 128


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        required=True,
        type=str.upper,  # accept any casing, e.g. --model ModernBERT
        choices=sorted(Checkpoint.__members__),
        help="which model",
    )
    p.add_argument("--text", required=True, nargs="+", help="one or more sentences")
    return p.parse_args(argv)


def resolve_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@torch.no_grad()
def predict(model, claims):
    """Classify sentences as claims.

    ``model`` is "BERT" or "MODERNBERT"; ``claims`` is a list of sentences.
    Returns one dict per sentence:
    ``{"text": ..., "label": "claim" | "not_claim", "prob_claim": float}``.
    """
    model = model.upper()
    ckpt = os.path.join(REPO_ROOT, Checkpoint[model])
    if not os.path.isdir(ckpt):
        raise SystemExit(f"checkpoint not found: {ckpt}\nTrain the model first, or fix Checkpoint.")

    # ModernBERT's torch.compile path is unsupported on MPS; sdpa replaces
    # flash-attention, which is CUDA-only (see src/models.py).
    kwargs = {"attn_implementation": "sdpa", "reference_compile": False}
    encoder = AutoModelForSequenceClassification.from_pretrained(
        ckpt, **(kwargs if model.upper() == "MODERNBERT" else {})
    )
    tokenizer = AutoTokenizer.from_pretrained(ckpt)

    device = resolve_device()
    encoder.eval().to(device)

    enc = tokenizer(
        claims, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt"
    ).to(device)
    probs = softmax(encoder(**enc).logits.float().cpu().numpy())

    return [
        {
            "text": text,
            "label": encoder.config.id2label[int(p.argmax())],
            "prob_claim": float(p[1]),
        }
        for text, p in zip(claims, probs)
    ]


def main(argv=None) -> int:
    args = parse_args(argv)
    for r in predict(args.model, args.text):
        print(f"{r['label']}  p(claim)={r['prob_claim']:.4f}  {r['text']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
