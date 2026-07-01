from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .data_loader import CANONICAL_LABEL, CANONICAL_TEXT
from .evaluate import compute_binary_metrics, save_confusion_matrix


def build_logistic_regression_pipeline(
    max_features: int,
    min_df: int,
    ngram_max: int,
    random_state: int,
) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=(1, ngram_max),
                    lowercase=True,
                    min_df=min_df,
                    max_df=0.95,
                ),
            ),
            (
                "classifier",
                LogisticRegression(max_iter=1000, random_state=random_state),
            ),
        ]
    )


def build_linear_svm_pipeline(max_features: int, min_df: int, ngram_max: int) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=(1, ngram_max),
                    lowercase=True,
                    min_df=min_df,
                    max_df=0.95,
                ),
            ),
            ("classifier", LinearSVC()),
        ]
    )


def _score_label_0_from_logistic(pipeline: Pipeline, texts: pd.Series) -> np.ndarray:
    classifier = pipeline.named_steps["classifier"]
    classes = list(classifier.classes_)
    label_0_index = classes.index(0)
    return pipeline.predict_proba(texts)[:, label_0_index]


def _score_label_0_from_svm(pipeline: Pipeline, texts: pd.Series) -> np.ndarray:
    classifier = pipeline.named_steps["classifier"]
    decision = pipeline.decision_function(texts)
    classes = list(classifier.classes_)
    if len(classes) != 2:
        raise ValueError(f"Expected binary SVM classes, got: {classes}")
    # For binary LinearSVC, positive decision favors classes_[1].
    return -decision if classes[1] == 1 else decision


def train_and_evaluate_baselines(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    models_dir: Path,
    figures_dir: Path,
    max_features: int,
    min_df: int,
    ngram_max: int,
    random_state: int,
) -> list[dict[str, float | str]]:
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics: list[dict[str, float | str]] = []
    x_train = train_df[CANONICAL_TEXT]
    y_train = train_df[CANONICAL_LABEL]
    x_test = test_df[CANONICAL_TEXT]
    y_test = test_df[CANONICAL_LABEL].to_numpy()

    logistic = build_logistic_regression_pipeline(
        max_features=max_features,
        min_df=min_df,
        ngram_max=ngram_max,
        random_state=random_state,
    )
    logistic.fit(x_train, y_train)
    logistic_pred = logistic.predict(x_test)
    logistic_score_0 = _score_label_0_from_logistic(logistic, x_test)
    metrics.append(
        compute_binary_metrics(y_test, logistic_pred, logistic_score_0, "tfidf_logistic_regression")
    )
    save_confusion_matrix(
        y_test,
        logistic_pred,
        "TF-IDF + Regressao Logistica",
        figures_dir / "confusion_matrix_logistic_regression.png",
    )
    joblib.dump(logistic, models_dir / "tfidf_logistic_regression.joblib")

    svm = build_linear_svm_pipeline(max_features=max_features, min_df=min_df, ngram_max=ngram_max)
    svm.fit(x_train, y_train)
    svm_pred = svm.predict(x_test)
    svm_score_0 = _score_label_0_from_svm(svm, x_test)
    metrics.append(compute_binary_metrics(y_test, svm_pred, svm_score_0, "tfidf_linear_svm"))
    save_confusion_matrix(
        y_test,
        svm_pred,
        "TF-IDF + SVM Linear",
        figures_dir / "confusion_matrix_linear_svm.png",
    )
    joblib.dump(svm, models_dir / "tfidf_linear_svm.joblib")

    return metrics

