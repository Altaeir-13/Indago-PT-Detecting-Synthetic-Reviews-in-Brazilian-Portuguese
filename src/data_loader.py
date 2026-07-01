from __future__ import annotations

import io
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from .config import DATASET_FILENAME, DATASET_URL


CANONICAL_TEXT = "text"
CANONICAL_CATEGORY = "category"
CANONICAL_LABEL = "label"


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the expected dataset file is not available locally."""


@dataclass(frozen=True, slots=True)
class ColumnMapping:
    text: str
    category: str
    label: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LoadedDataset:
    frame: pd.DataFrame
    mapping: ColumnMapping


def dataset_missing_message(path: Path) -> str:
    return (
        f"Dataset not found at: {path}\n"
        f"Download {DATASET_FILENAME} from {DATASET_URL} and place it at:\n"
        f"  data/raw/{DATASET_FILENAME}\n"
        "Then run the experiment again. No metrics were generated."
    )


def _normalize_name(name: str) -> str:
    name = str(name).strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", name).strip("_")


def _find_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    normalized = {_normalize_name(col): col for col in columns}
    for candidate in candidates:
        key = _normalize_name(candidate)
        if key in normalized:
            return normalized[key]
    for col in columns:
        key = _normalize_name(col)
        if any(candidate in key for candidate in candidates):
            return col
    return None


def infer_columns(df: pd.DataFrame) -> ColumnMapping:
    columns = list(df.columns)
    text = _find_column(
        columns,
        (
            "review_comment_message",
            "review_text",
            "review",
            "text",
            "texto",
            "comentario",
            "mensagem",
        ),
    )
    category = _find_column(
        columns,
        (
            "product_category_name",
            "category",
            "categoria",
            "product_category",
            "classe_produto",
        ),
    )
    label = _find_column(
        columns,
        (
            "label",
            "target",
            "class",
            "classe",
            "rotulo",
            "fake",
            "is_fake",
        ),
    )

    missing = [
        name
        for name, value in (("text", text), ("category", category), ("label", label))
        if value is None
    ]
    if missing:
        raise ValueError(
            "Could not infer required columns "
            f"{missing}. Available columns: {columns}. "
            "Rename the CSV columns or update data_loader.py."
        )
    return ColumnMapping(text=text, category=category, label=label)


def _read_csv(path: Path) -> pd.DataFrame:
    attempts = (
        {"engine": "python"},
        {"engine": "python", "on_bad_lines": "warn"},
        {"engine": "python", "sep": ","},
    )
    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            df = pd.read_csv(path, **kwargs)
            if df.shape[1] >= 3:
                return df
        except Exception as exc:  # pragma: no cover - exercised by malformed files.
            last_error = exc

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        return pd.read_csv(io.StringIO(normalized), engine="python")
    except Exception as exc:
        if last_error is not None:
            raise ValueError(f"Could not read CSV. Last parser error: {last_error}") from exc
        raise


def _coerce_labels(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        labels = pd.to_numeric(series, errors="coerce")
    else:
        mapping = {
            "0": 0,
            "fake": 0,
            "false": 0,
            "falsa": 0,
            "sintetica": 0,
            "synthetic": 0,
            "1": 1,
            "true": 1,
            "real": 1,
            "genuine": 1,
            "genuina": 1,
            "verdadeira": 1,
        }
        labels = series.astype(str).str.strip().str.lower().map(mapping)

    if labels.isna().any():
        bad_values = series[labels.isna()].drop_duplicates().head(10).tolist()
        raise ValueError(f"Could not map label values to 0/1. Examples: {bad_values}")

    labels = labels.astype(int)
    values = set(labels.unique().tolist())
    if not values.issubset({0, 1}):
        raise ValueError(f"Labels must be binary 0/1. Found values: {sorted(values)}")
    return labels


def load_dataset(path: str | Path) -> LoadedDataset:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise DatasetNotFoundError(dataset_missing_message(dataset_path))

    raw = _read_csv(dataset_path)
    mapping = infer_columns(raw)

    frame = pd.DataFrame(
        {
            CANONICAL_TEXT: raw[mapping.text],
            CANONICAL_CATEGORY: raw[mapping.category],
            CANONICAL_LABEL: _coerce_labels(raw[mapping.label]),
        }
    )
    return LoadedDataset(frame=frame, mapping=mapping)

