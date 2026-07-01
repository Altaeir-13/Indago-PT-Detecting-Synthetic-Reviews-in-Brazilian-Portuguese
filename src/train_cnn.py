from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .data_loader import CANONICAL_LABEL, CANONICAL_TEXT
from .utils import get_device, set_seed


TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


@dataclass(frozen=True, slots=True)
class CNNHyperparameters:
    vocab_size: int
    max_len: int
    embedding_dim: int
    filters: int
    kernel_size: int
    dropout: float
    learning_rate: float
    batch_size: int
    epochs: int
    patience: int


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text).lower())


def build_vocab(texts: pd.Series, vocab_size: int) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokenize(text))

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, _count in counter.most_common(max(0, vocab_size - len(vocab))):
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def encode_text(text: str, vocab: dict[str, int], max_len: int) -> list[int]:
    ids = [vocab.get(token, vocab[UNK_TOKEN]) for token in tokenize(text)]
    ids = ids[:max_len]
    if len(ids) < max_len:
        ids.extend([vocab[PAD_TOKEN]] * (max_len - len(ids)))
    return ids


class ReviewDataset(Dataset):
    def __init__(
        self,
        texts: pd.Series | list[str],
        labels: pd.Series | list[int] | None,
        vocab: dict[str, int],
        max_len: int,
    ) -> None:
        self.inputs = torch.tensor([encode_text(text, vocab, max_len) for text in texts], dtype=torch.long)
        if labels is None:
            self.labels = None
        else:
            self.labels = torch.tensor(np.asarray(labels, dtype=np.float32), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        if self.labels is None:
            return self.inputs[index]
        return self.inputs[index], self.labels[index]


class CNN1DClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        filters: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=filters,
            kernel_size=kernel_size,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(filters, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        channels_first = embedded.transpose(1, 2)
        features = self.relu(self.conv(channels_first))
        pooled = torch.max(features, dim=2).values
        dropped = self.dropout(pooled)
        return self.classifier(dropped).squeeze(1)


def _accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= 0.5).float()
    return float((predictions == labels).float().mean().item())


def _run_epoch(
    model: CNN1DClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)
    losses: list[float] = []
    accuracies: list[float] = []

    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(is_training):
            logits = model(input_ids)
            loss = criterion(logits, labels)
            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        batch_size = labels.shape[0]
        losses.append(float(loss.item()) * batch_size)
        accuracies.append(_accuracy_from_logits(logits.detach(), labels) * batch_size)

    total = len(loader.dataset)
    return sum(losses) / total, sum(accuracies) / total


def train_cnn(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    hparams: CNNHyperparameters,
    model_path: Path,
    random_state: int = 42,
    device: torch.device | None = None,
) -> tuple[CNN1DClassifier, dict[str, int], pd.DataFrame]:
    if hparams.max_len < hparams.kernel_size:
        raise ValueError("max_len must be greater than or equal to kernel_size.")

    set_seed(random_state)
    device = device or get_device()
    vocab = build_vocab(train_df[CANONICAL_TEXT], hparams.vocab_size)
    model = CNN1DClassifier(
        vocab_size=hparams.vocab_size,
        embedding_dim=hparams.embedding_dim,
        filters=hparams.filters,
        kernel_size=hparams.kernel_size,
        dropout=hparams.dropout,
    ).to(device)

    train_dataset = ReviewDataset(
        train_df[CANONICAL_TEXT], train_df[CANONICAL_LABEL], vocab, hparams.max_len
    )
    val_dataset = ReviewDataset(val_df[CANONICAL_TEXT], val_df[CANONICAL_LABEL], vocab, hparams.max_len)
    train_loader = DataLoader(train_dataset, batch_size=hparams.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=hparams.batch_size, shuffle=False)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.learning_rate)

    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    wait = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, hparams.epochs + 1):
        train_loss, train_accuracy = _run_epoch(model, train_loader, criterion, device, optimizer)
        with torch.no_grad():
            val_loss, val_accuracy = _run_epoch(model, val_loader, criterion, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": train_accuracy,
                "val_accuracy": val_accuracy,
            }
        )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            wait = 0
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": best_state,
                    "vocab": vocab,
                    "hyperparameters": asdict(hparams),
                    "best_val_loss": best_val_loss,
                    "epoch": epoch,
                },
                model_path,
            )
        else:
            wait += 1
            if wait >= hparams.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, vocab, pd.DataFrame(history)


def predict_cnn(
    model: CNN1DClassifier,
    vocab: dict[str, int],
    texts: pd.Series | list[str],
    max_len: int,
    batch_size: int,
    device: torch.device | None = None,
) -> dict[str, np.ndarray]:
    device = device or get_device()
    model = model.to(device)
    model.eval()
    dataset = ReviewDataset(texts, labels=None, vocab=vocab, max_len=max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    logits_batches: list[np.ndarray] = []
    with torch.no_grad():
        for input_ids in loader:
            input_ids = input_ids.to(device)
            logits = model(input_ids)
            logits_batches.append(logits.detach().cpu().numpy())

    logits = np.concatenate(logits_batches)
    prob_label_1 = 1.0 / (1.0 + np.exp(-logits))
    prob_label_0 = 1.0 - prob_label_1
    pred_label = (prob_label_1 >= 0.5).astype(int)
    return {
        "logits": logits,
        "prob_label_1": prob_label_1,
        "prob_label_0": prob_label_0,
        "pred_label": pred_label,
    }

