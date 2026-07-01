from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .data_loader import CANONICAL_CATEGORY, CANONICAL_LABEL, CANONICAL_TEXT
from .utils import save_json


def _save_barplot(table: pd.DataFrame, x: str, y: str, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=table, x=x, y=y, ax=ax)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_eda(df: pd.DataFrame, tables_dir: Path, figures_dir: Path, random_state: int = 42) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "total_samples_after_cleaning": int(len(df)),
        "missing_text": int(df[CANONICAL_TEXT].isna().sum()),
        "missing_category": int(df[CANONICAL_CATEGORY].isna().sum()),
        "missing_label": int(df[CANONICAL_LABEL].isna().sum()),
        "duplicated_rows": int(df.duplicated().sum()),
        "duplicated_texts": int(df[CANONICAL_TEXT].duplicated().sum()),
        "char_len_min": int(df["char_len"].min()),
        "char_len_mean": float(df["char_len"].mean()),
        "char_len_median": float(df["char_len"].median()),
        "char_len_max": int(df["char_len"].max()),
        "token_len_min": int(df["token_len"].min()),
        "token_len_mean": float(df["token_len"].mean()),
        "token_len_median": float(df["token_len"].median()),
        "token_len_max": int(df["token_len"].max()),
    }
    save_json(summary, tables_dir / "eda_summary.json")

    missing = df.isna().sum().reset_index()
    missing.columns = ["column", "missing_count"]
    missing.to_csv(tables_dir / "missing_values.csv", index=False)

    class_dist = (
        df[CANONICAL_LABEL]
        .value_counts()
        .sort_index()
        .rename_axis(CANONICAL_LABEL)
        .reset_index(name="count")
    )
    class_dist["class_name"] = class_dist[CANONICAL_LABEL].map(
        {0: "falsa/sintetica", 1: "genuina"}
    )
    class_dist.to_csv(tables_dir / "class_distribution.csv", index=False)
    _save_barplot(
        class_dist,
        "class_name",
        "count",
        "Distribuicao por classe",
        figures_dir / "class_distribution.png",
    )

    category_dist = (
        df[CANONICAL_CATEGORY]
        .value_counts()
        .rename_axis(CANONICAL_CATEGORY)
        .reset_index(name="count")
    )
    category_dist.to_csv(tables_dir / "category_distribution.csv", index=False)
    _save_barplot(
        category_dist,
        CANONICAL_CATEGORY,
        "count",
        "Distribuicao por categoria",
        figures_dir / "category_distribution.png",
    )

    length_summary = df[["char_len", "token_len"]].describe().T.reset_index()
    length_summary = length_summary.rename(columns={"index": "metric"})
    length_summary.to_csv(tables_dir / "review_length_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["char_len"], bins=40, kde=False, ax=ax)
    ax.set_title("Distribuicao do tamanho das reviews em caracteres")
    ax.set_xlabel("Caracteres")
    fig.tight_layout()
    fig.savefig(figures_dir / "review_char_length_hist.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["token_len"], bins=40, kde=False, ax=ax)
    ax.set_title("Distribuicao do tamanho das reviews em tokens")
    ax.set_xlabel("Tokens por espaco")
    fig.tight_layout()
    fig.savefig(figures_dir / "review_token_length_hist.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=df, x=CANONICAL_LABEL, y="token_len", ax=ax)
    ax.set_title("Tamanho das reviews por classe")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Tokens")
    fig.tight_layout()
    fig.savefig(figures_dir / "review_token_length_by_class.png", dpi=160)
    plt.close(fig)

    examples = (
        df.groupby(CANONICAL_LABEL, group_keys=False)
        .sample(n=min(5, df[CANONICAL_LABEL].value_counts().min()), random_state=random_state)
        [[CANONICAL_LABEL, CANONICAL_CATEGORY, CANONICAL_TEXT]]
    )
    examples.to_csv(tables_dir / "examples_by_class.csv", index=False)

