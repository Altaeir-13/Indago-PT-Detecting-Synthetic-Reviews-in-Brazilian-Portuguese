from __future__ import annotations

import re
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from .data_loader import CANONICAL_CATEGORY, CANONICAL_LABEL, CANONICAL_TEXT


SPACE_RE = re.compile(r"\s+")


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return SPACE_RE.sub(" ", text).strip()


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned[CANONICAL_TEXT] = cleaned[CANONICAL_TEXT].map(normalize_text)
    cleaned[CANONICAL_CATEGORY] = (
        cleaned[CANONICAL_CATEGORY].fillna("unknown").astype(str).map(normalize_text)
    )
    cleaned = cleaned[cleaned[CANONICAL_TEXT].str.len() > 0].copy()
    cleaned[CANONICAL_LABEL] = cleaned[CANONICAL_LABEL].astype(int)
    cleaned["char_len"] = cleaned[CANONICAL_TEXT].str.len()
    cleaned["token_len"] = cleaned[CANONICAL_TEXT].str.split().str.len()
    return cleaned.reset_index(drop=True)


def split_dataset(
    df: pd.DataFrame,
    random_state: int = 42,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if abs((train_size + val_size + test_size) - 1.0) > 1e-8:
        raise ValueError("train_size + val_size + test_size must sum to 1.0")

    labels = df[CANONICAL_LABEL]
    if labels.nunique() != 2:
        raise ValueError("The experiment requires both classes 0 and 1.")

    temp_size = val_size + test_size
    try:
        train_df, temp_df = train_test_split(
            df,
            train_size=train_size,
            random_state=random_state,
            stratify=labels,
        )
        relative_test_size = test_size / temp_size
        val_df, test_df = train_test_split(
            temp_df,
            test_size=relative_test_size,
            random_state=random_state,
            stratify=temp_df[CANONICAL_LABEL],
        )
    except ValueError as exc:
        raise ValueError(
            "Could not create a stratified 70/15/15 split. "
            "Check that each class has enough examples."
        ) from exc

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )

