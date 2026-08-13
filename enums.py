"""Shared enumerations."""

from enum import Enum


class Checkpoint(str, Enum):
    """Fine-tuned checkpoints, by model name, as paths relative to the repo root."""

    BERT = "results/runs/bert-base-uncased_lr5e-05_e5_s42/checkpoint-2905"
    MODERNBERT = "results/runs/answerdotai_ModernBERT-base_lr5e-05_e5_s42/checkpoint-1743"
