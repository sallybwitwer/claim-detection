"""Model and tokenizer construction for the encoder arm."""

from typing import Any, Dict, Tuple

from transformers import AutoModelForSequenceClassification, AutoTokenizer

ID2LABEL = {0: "not_claim", 1: "claim"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


def is_modernbert(model_name: str) -> bool:
    return "modernbert" in model_name.lower()


def model_slug(model_name: str) -> str:
    """``answerdotai/ModernBERT-base`` -> ``answerdotai_ModernBERT-base``."""
    return model_name.replace("/", "_")


def build_model_and_tokenizer(
    model_name: str, num_labels: int = 2
) -> Tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    kwargs: Dict[str, Any] = {
        "num_labels": num_labels,
        "id2label": ID2LABEL,
        "label2id": LABEL2ID,
    }
    if is_modernbert(model_name):
        # flash-attention is CUDA-only, and ModernBERT's default torch.compile
        # path is unsupported on the MPS backend.
        kwargs["attn_implementation"] = "sdpa"
        kwargs["reference_compile"] = False

    model = AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)
    return model, tokenizer
