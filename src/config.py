from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ExperimentConfig:
    data_path: Path = Path("data/raw/true_fake_dataset_top15.csv")
    output_dir: Path = Path("outputs")
    random_state: int = 42

    train_size: float = 0.70
    val_size: float = 0.15
    test_size: float = 0.15

    text_column: str = "text"
    category_column: str = "category"
    label_column: str = "label"

    vocab_size: int = 20_000
    max_len: int = 128
    embedding_dim: int = 128
    filters: int = 128
    kernel_size: int = 5
    dropout: float = 0.5
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 30
    patience: int = 4

    tfidf_max_features: int = 20_000
    tfidf_min_df: int = 2
    tfidf_ngram_max: int = 2

    error_samples: int = 10

    @property
    def figures_dir(self) -> Path:
        return self.output_dir / "figures"

    @property
    def tables_dir(self) -> Path:
        return self.output_dir / "tables"

    @property
    def models_dir(self) -> Path:
        return self.output_dir / "models"

    @property
    def cnn_model_path(self) -> Path:
        return self.models_dir / "cnn1d.pt"


DATASET_URL = "https://github.com/cristianomg10/fake-reviews-ptbr-dataset"
DATASET_FILENAME = "true_fake_dataset_top15.csv"

