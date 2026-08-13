"""Fine-tune an encoder for claim detection on data/ours.

    python scripts/finetune.py --model bert-base-uncased --lr 5e-5
    python scripts/finetune.py --model answerdotai/ModernBERT-base --lr 5e-5

Checkpoints are selected on a held-out dev split carved from train; the test
split is scored exactly once, at the end, with the best checkpoint.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import torch
import transformers
from transformers import (
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import clean_train, load_splits, make_dev_split, to_dataset  # noqa: E402
from src.metrics import compute_metrics, predictions_frame, scores  # noqa: E402
from src.models import build_model_and_tokenizer, model_slug  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="HF model id")
    p.add_argument("--lr", type=float, default=5e-5) # learning rate - same as paper
    p.add_argument("--epochs", type=float, default=5) # num training rounds
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dev-size", type=float, default=0.1)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--data-dir", default="data/ours")
    p.add_argument("--output-root", default="results")
    p.add_argument("--run-name", default=None, help="defaults to {model}_lr{lr}_s{seed}")
    p.add_argument("--tensorboard", action="store_true", help="requires tensorboard")
    p.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="truncate train for a quick smoke test",
    )
    return p.parse_args(argv)


def resolve_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main(argv=None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)

    run_name = args.run_name or f"{model_slug(args.model)}_lr{args.lr:g}_s{args.seed}"
    run_dir = os.path.join(args.output_root, "runs", run_name)
    preds_dir = os.path.join(args.output_root, "preds")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(preds_dir, exist_ok=True)

    train_raw, test_df = load_splits(args.data_dir)
    train_clean, clean_report = clean_train(train_raw, test_df)
    train_df, dev_df = make_dev_split(train_clean, args.dev_size, args.seed)
    if args.max_train_samples:
        train_df = train_df.head(args.max_train_samples)

    print(
        f"[data] train={len(train_df)} dev={len(dev_df)} test={len(test_df)} "
        f"| cleaning: {clean_report}"
    )

    model, tokenizer = build_model_and_tokenizer(args.model)
    train_ds = to_dataset(train_df, tokenizer, args.max_length)
    dev_ds = to_dataset(dev_df, tokenizer, args.max_length)
    test_ds = to_dataset(test_df, tokenizer, args.max_length)

    device = resolve_device()
    print(f"[env] device={device} torch={torch.__version__} tf={transformers.__version__}")

    training_args = TrainingArguments(
        output_dir=run_dir,
        run_name=run_name,
        seed=args.seed,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        # Mixed precision is unreliable on MPS; these models are small enough
        # that fp32 costs little.
        fp16=False,
        bf16=False,
        dataloader_num_workers=0,
        report_to=["tensorboard"] if args.tensorboard else "none",
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    dev_metrics = trainer.evaluate(dev_ds, metric_key_prefix="dev")
    test_output = trainer.predict(test_ds, metric_key_prefix="test")
    test_logits = test_output.predictions
    if isinstance(test_logits, tuple):
        test_logits = test_logits[0]
    test_logits = np.asarray(test_logits)
    test_metrics = scores(test_df["label"].to_numpy(), test_logits.argmax(axis=-1))

    preds_path = os.path.join(preds_dir, f"{run_name}.csv")
    predictions_frame(test_df["text"], test_df["label"], test_logits).to_csv(
        preds_path, index=False
    )

    metrics = {"run": run_name, "model": args.model, "dev": dev_metrics, "test": test_metrics}
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    manifest = {
        "run": run_name,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "device": device,
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "data": {
            "cleaning": clean_report,
            "n_train": len(train_df),
            "n_dev": len(dev_df),
            "n_test": len(test_df),
        },
        "preds_path": preds_path,
    }
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[dev ] {json.dumps({k: round(v, 4) for k, v in dev_metrics.items() if isinstance(v, float)})}")
    print(f"[test] {json.dumps({k: round(v, 4) for k, v in test_metrics.items()})}")
    print(f"[out ] {run_dir}  |  {preds_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
