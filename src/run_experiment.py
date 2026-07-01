from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .baselines import train_and_evaluate_baselines
from .config import ExperimentConfig
from .data_loader import DatasetNotFoundError, load_dataset
from .eda import run_eda
from .evaluate import (
    build_error_analysis,
    compute_binary_metrics,
    save_confusion_matrix,
    save_model_comparison,
    save_training_history,
)
from .preprocess import clean_dataframe, split_dataset
from .train_cnn import CNNHyperparameters, predict_cnn, train_cnn
from .utils import ensure_directories, get_device, print_path_summary, save_json, set_seed


def build_arg_parser() -> argparse.ArgumentParser:
    defaults = ExperimentConfig()
    parser = argparse.ArgumentParser(
        description="Run the Fake Reviews PT-BR experiment with TF-IDF baselines and a PyTorch CNN 1D."
    )
    parser.add_argument("--data-path", type=Path, default=defaults.data_path)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--random-state", type=int, default=defaults.random_state)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--vocab-size", type=int, default=defaults.vocab_size)
    parser.add_argument("--max-len", type=int, default=defaults.max_len)
    parser.add_argument("--embedding-dim", type=int, default=defaults.embedding_dim)
    parser.add_argument("--filters", type=int, default=defaults.filters)
    parser.add_argument("--kernel-size", type=int, default=defaults.kernel_size)
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument("--tfidf-max-features", type=int, default=defaults.tfidf_max_features)
    parser.add_argument("--tfidf-min-df", type=int, default=defaults.tfidf_min_df)
    parser.add_argument("--tfidf-ngram-max", type=int, default=defaults.tfidf_ngram_max)
    parser.add_argument("--error-samples", type=int, default=defaults.error_samples)
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-cnn", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        random_state=args.random_state,
        vocab_size=args.vocab_size,
        max_len=args.max_len,
        embedding_dim=args.embedding_dim,
        filters=args.filters,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        tfidf_max_features=args.tfidf_max_features,
        tfidf_min_df=args.tfidf_min_df,
        tfidf_ngram_max=args.tfidf_ngram_max,
        error_samples=args.error_samples,
    )


def run_experiment(config: ExperimentConfig, skip_baselines: bool = False, skip_cnn: bool = False) -> pd.DataFrame:
    ensure_directories(config)
    set_seed(config.random_state)

    loaded = load_dataset(config.data_path)
    save_json(loaded.mapping.to_dict(), config.tables_dir / "column_mapping.json")

    df = clean_dataframe(loaded.frame)
    run_eda(df, config.tables_dir, config.figures_dir, random_state=config.random_state)

    train_df, val_df, test_df = split_dataset(
        df,
        random_state=config.random_state,
        train_size=config.train_size,
        val_size=config.val_size,
        test_size=config.test_size,
    )
    train_df.to_csv(config.tables_dir / "train_split_preview.csv", index=False)
    val_df.to_csv(config.tables_dir / "validation_split_preview.csv", index=False)
    test_df.to_csv(config.tables_dir / "test_split_preview.csv", index=False)

    metrics: list[dict[str, float | str]] = []

    if not skip_baselines:
        metrics.extend(
            train_and_evaluate_baselines(
                train_df=train_df,
                test_df=test_df,
                models_dir=config.models_dir,
                figures_dir=config.figures_dir,
                max_features=config.tfidf_max_features,
                min_df=config.tfidf_min_df,
                ngram_max=config.tfidf_ngram_max,
                random_state=config.random_state,
            )
        )

    if not skip_cnn:
        hparams = CNNHyperparameters(
            vocab_size=config.vocab_size,
            max_len=config.max_len,
            embedding_dim=config.embedding_dim,
            filters=config.filters,
            kernel_size=config.kernel_size,
            dropout=config.dropout,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            epochs=config.epochs,
            patience=config.patience,
        )
        device = get_device()
        model, vocab, history = train_cnn(
            train_df=train_df,
            val_df=val_df,
            hparams=hparams,
            model_path=config.cnn_model_path,
            random_state=config.random_state,
            device=device,
        )
        save_training_history(history, config.figures_dir, config.tables_dir)
        predictions = predict_cnn(
            model=model,
            vocab=vocab,
            texts=test_df[config.text_column],
            max_len=config.max_len,
            batch_size=config.batch_size,
            device=device,
        )
        y_test = test_df[config.label_column].to_numpy()
        metrics.append(
            compute_binary_metrics(
                y_true=y_test,
                y_pred=predictions["pred_label"],
                score_label_0=predictions["prob_label_0"],
                model_name="cnn1d_pytorch",
            )
        )
        save_confusion_matrix(
            y_test,
            predictions["pred_label"],
            "CNN 1D PyTorch",
            config.figures_dir / "confusion_matrix_cnn1d.png",
        )
        errors = build_error_analysis(
            test_df=test_df,
            y_pred=predictions["pred_label"],
            score_label_0=predictions["prob_label_0"],
            n_samples=config.error_samples,
        )
        errors.to_csv(config.tables_dir / "cnn_error_analysis.csv", index=False)

    comparison = save_model_comparison(metrics, config.tables_dir)
    print_path_summary(config)
    print(comparison.to_string(index=False))
    return comparison


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args)

    try:
        run_experiment(config, skip_baselines=args.skip_baselines, skip_cnn=args.skip_cnn)
    except DatasetNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

