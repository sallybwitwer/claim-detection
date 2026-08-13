"""Loading, cleaning and tokenizing the claim-detection splits.

The cleaning step is shared by every arm of the comparison (encoders now, the
LLM arm later), so the exact rows that get dropped are reported back to the
caller for the run manifest rather than just logged.
"""

from typing import Dict, Optional, Tuple

import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split

TEXT_COL = "text"
LABEL_COL = "label"


def load_splits(data_dir: str = "data/ours") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read train.csv and test.csv from ``data_dir``."""
    train = pd.read_csv(f"{data_dir}/train.csv")
    test = pd.read_csv(f"{data_dir}/test.csv")
    for name, df in (("train", train), ("test", test)):
        missing = {TEXT_COL, LABEL_COL} - set(df.columns)
        if missing:
            raise ValueError(f"{name}.csv is missing column(s): {sorted(missing)}")
    return train, test


def clean_train(
    train: pd.DataFrame, test: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Drop empty, duplicated and test-leaking rows from train.

    ``test`` is only read, never modified -- the test split is treated as fixed.
    """
    report = {"rows_in": len(train)}
    df = train.copy()

    blank = df[TEXT_COL].isna() | (df[TEXT_COL].astype(str).str.strip() == "")
    report["dropped_empty_text"] = int(blank.sum())
    df = df[~blank]

    dupes = df.duplicated(subset=[TEXT_COL], keep="first")
    report["dropped_duplicate_text"] = int(dupes.sum())
    df = df[~dupes]

    leaked = df[TEXT_COL].isin(set(test[TEXT_COL].dropna()))
    report["dropped_overlaps_test"] = int(leaked.sum())
    df = df[~leaked]

    df = df.reset_index(drop=True)
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    report["rows_out"] = len(df)
    return df, report


def make_dev_split(
    train: pd.DataFrame, dev_size: float = 0.1, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/dev split so checkpoint selection never touches test."""
    tr, dev = train_test_split(
        train,
        test_size=dev_size,
        random_state=seed,
        stratify=train[LABEL_COL],
    )
    return tr.reset_index(drop=True), dev.reset_index(drop=True)


def to_dataset(
    df: pd.DataFrame, tokenizer, max_length: int = 128, num_proc: Optional[int] = None
) -> Dataset:
    """Tokenize a frame into a ``Dataset`` ready for ``Trainer``.

    No padding here -- the collator pads each batch to its own longest sequence,
    which is meaningfully faster than padding everything to ``max_length``.
    """
    ds = Dataset.from_pandas(df[[TEXT_COL, LABEL_COL]], preserve_index=False)

    def encode(batch):
        return tokenizer(batch[TEXT_COL], truncation=True, max_length=max_length)

    ds = ds.map(encode, batched=True, num_proc=num_proc, remove_columns=[TEXT_COL])
    return ds.rename_column(LABEL_COL, "labels")
