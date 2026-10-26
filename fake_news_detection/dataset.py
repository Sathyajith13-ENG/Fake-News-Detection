"""CSV loading and validation."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

VALID_LABELS = ("FAKE", "REAL")
MIN_PER_CLASS = 2


class DatasetError(ValueError):
    """Raised when the dataset cannot be used for training."""


def validate_frame(df: pd.DataFrame, text_column: str = "text", label_column: str = "label") -> pd.DataFrame:
    missing = [c for c in (text_column, label_column) if c not in df.columns]
    if missing:
        raise DatasetError(
            f"Missing column(s): {', '.join(missing)}. Found: {', '.join(map(str, df.columns))}. "
            "Use --text-column / --label-column to name your columns."
        )
    out = pd.DataFrame(
        {
            "text": df[text_column].astype("string").fillna("").str.strip(),
            "label": df[label_column].astype("string").fillna("").str.strip().str.upper(),
        }
    )
    out = out[out["text"] != ""]
    bad = sorted(set(out["label"]) - set(VALID_LABELS))
    if bad:
        raise DatasetError(f"Invalid label(s) {bad}. Labels must be FAKE or REAL (case-insensitive).")
    counts = out["label"].value_counts()
    for label in VALID_LABELS:
        if int(counts.get(label, 0)) < MIN_PER_CLASS:
            raise DatasetError(
                f"Need at least {MIN_PER_CLASS} {label} examples with non-empty text; "
                f"found {int(counts.get(label, 0))}."
            )
    return out.reset_index(drop=True)


def load_dataset(path: str | Path, text_column: str = "text", label_column: str = "label") -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise DatasetError(f"Dataset not found: {path}. Put a CSV with text,label columns there or pass --data.")
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise DatasetError(f"Could not read {path} as a UTF-8 CSV: {exc}") from exc
    return validate_frame(df, text_column, label_column)
