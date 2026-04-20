#!/usr/bin/env python3
"""
CSV-only baseline for GolfDB club classification.

This script intentionally ignores the videos and trains a RandomForest model
using only metadata columns from GolfDB.csv.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.preprocessing import LabelEncoder


warnings.filterwarnings("ignore")


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "GolfDB.csv"
OUTPUT_DIR = BASE_DIR / "model_results" / "csv_baseline"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CSV-only GolfDB baseline.")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Maximum metadata rows to use. Use 0 for all rows.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Folder where plots are saved.",
    )
    return parser.parse_args()


def sample_balanced_rows(dataframe: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    clubs = sorted(dataframe["club"].dropna().unique())
    per_club = max(1, max_rows // len(clubs))
    selected_parts = []
    selected_indexes: set[int] = set()

    for club in clubs:
        group = dataframe[dataframe["club"] == club]
        take = min(len(group), per_club)
        sampled = group.sample(n=take, random_state=RANDOM_STATE)
        selected_parts.append(sampled)
        selected_indexes.update(sampled.index.tolist())

    selected = pd.concat(selected_parts, axis=0)
    remaining = max_rows - len(selected)

    if remaining > 0:
        leftovers = dataframe.drop(index=list(selected_indexes))
        extra = leftovers.sample(
            n=min(remaining, len(leftovers)),
            random_state=RANDOM_STATE,
        )
        selected = pd.concat([selected, extra], axis=0)

    return selected.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def load_data(max_rows: int) -> pd.DataFrame:
    print("Loading GolfDB.csv...")
    dataframe = pd.read_csv(CSV_PATH)
    dataframe = dataframe.loc[:, ~dataframe.columns.str.startswith("Unnamed")]

    if max_rows > 0 and len(dataframe) > max_rows:
        dataframe = sample_balanced_rows(dataframe, max_rows)

    club_counts = {
        club: int(count)
        for club, count in dataframe["club"].value_counts().sort_index().items()
    }
    print(f"   Rows selected: {len(dataframe)}")
    print(f"   Clubs: {club_counts}")
    return dataframe


def prepare_features(
    dataframe: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict[str, LabelEncoder], list[str]]:
    print("Preparing CSV features...")
    feature_columns = ["player", "sex", "view", "slow"]
    prepared = dataframe[feature_columns + ["club"]].copy()

    label_encoders: dict[str, LabelEncoder] = {}
    encoded_columns = []

    for column in ["player", "sex", "view"]:
        encoder = LabelEncoder()
        encoded_values = encoder.fit_transform(prepared[column].fillna("unknown").astype(str))
        label_encoders[column] = encoder
        encoded_columns.append(pd.Series(encoded_values, name=column))

    target_encoder = LabelEncoder()
    target = target_encoder.fit_transform(prepared["club"].astype(str))
    label_encoders["club"] = target_encoder

    numeric_columns = prepared[["slow"]].apply(pd.to_numeric, errors="coerce").fillna(0)
    feature_frame = pd.concat([*encoded_columns, numeric_columns], axis=1)
    feature_names = feature_frame.columns.tolist()

    print(f"   Feature matrix: {feature_frame.shape}")
    print(f"   Target classes: {list(target_encoder.classes_)}")
    return feature_frame.to_numpy(dtype=float), target, label_encoders, feature_names


def split_data(
    features: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    class_counts = pd.Series(target).value_counts()
    class_count = len(class_counts)
    can_stratify = class_count > 1 and class_counts.min() >= 2

    if can_stratify:
        test_count = max(math.ceil(len(target) * TEST_SIZE), class_count)
        test_count = min(test_count, len(target) - class_count)
        stratify = target
    else:
        test_count = max(1, math.ceil(len(target) * TEST_SIZE))
        stratify = None
        print("   Using non-stratified split because a class has too few rows.")

    return train_test_split(
        features,
        target,
        test_size=test_count,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )


def build_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=120,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def train_random_forest(
    train_features: np.ndarray,
    train_target: np.ndarray,
    feature_names: list[str],
) -> RandomForestClassifier:
    print("Training CSV-only RandomForest...")
    model = build_model()
    model.fit(train_features, train_target)

    print("   Feature importances:")
    for name, importance in sorted(
        zip(feature_names, model.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"      {name}: {importance:.4f}")

    return model


def evaluate_predictions(
    test_target: np.ndarray,
    predictions: np.ndarray,
    label_encoders: dict[str, LabelEncoder],
) -> dict[str, object]:
    class_names = label_encoders["club"].classes_
    labels = np.arange(len(class_names))
    accuracy = accuracy_score(test_target, predictions)

    print(f"   Test accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            test_target,
            predictions,
            labels=labels,
            target_names=class_names,
            zero_division=0,
        )
    )

    precision, recall, f1_score, support = precision_recall_fscore_support(
        test_target,
        predictions,
        labels=labels,
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1_score,
        "support": support,
        "test_target": test_target,
        "predictions": predictions,
        "labels": labels,
        "class_names": class_names,
    }


def plot_learning_curve(
    model: RandomForestClassifier,
    train_features: np.ndarray,
    train_target: np.ndarray,
    output_dir: Path,
) -> None:
    train_sizes, train_scores, validation_scores = learning_curve(
        model,
        train_features,
        train_target,
        cv=CV_FOLDS,
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1,
        scoring="accuracy",
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    validation_mean = np.mean(validation_scores, axis=1)
    validation_std = np.std(validation_scores, axis=1)

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(
        train_sizes,
        train_mean,
        "o-",
        color="#2E86AB",
        label="Training Accuracy",
        linewidth=2,
    )
    axis.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.2,
        color="#2E86AB",
    )
    axis.plot(
        train_sizes,
        validation_mean,
        "s-",
        color="#A23B72",
        label="Validation Accuracy",
        linewidth=2,
    )
    axis.fill_between(
        train_sizes,
        validation_mean - validation_std,
        validation_mean + validation_std,
        alpha=0.2,
        color="#A23B72",
    )
    axis.set_xlabel("Training Set Size", fontsize=12, fontweight="bold")
    axis.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
    axis.set_title("CSV Baseline Learning Curve", fontsize=14, fontweight="bold")
    axis.legend(fontsize=11, loc="lower right")
    axis.grid(True, alpha=0.3)
    axis.set_ylim([0.0, 1.05])
    figure.tight_layout()
    figure.savefig(output_dir / "learning_curve.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_confusion_matrix(metrics: dict[str, object], output_dir: Path) -> None:
    matrix = confusion_matrix(
        metrics["test_target"],
        metrics["predictions"],
        labels=metrics["labels"],
    )

    figure, axis = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=metrics["class_names"],
        yticklabels=metrics["class_names"],
        ax=axis,
        cbar_kws={"label": "Count"},
    )
    axis.set_xlabel("Prediction", fontsize=12, fontweight="bold")
    axis.set_ylabel("Actual", fontsize=12, fontweight="bold")
    axis.set_title("CSV Baseline Confusion Matrix", fontsize=14, fontweight="bold")
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_class_metrics(metrics: dict[str, object], output_dir: Path) -> None:
    metric_values = [
        ("Precision", metrics["precision"], "#2E86AB"),
        ("Recall", metrics["recall"], "#A23B72"),
        ("F1-Score", metrics["f1"], "#F18F01"),
    ]

    figure, axes = plt.subplots(1, 3, figsize=(15, 4))

    for axis, (title, values, color) in zip(axes, metric_values):
        axis.bar(metrics["class_names"], values, color=color, alpha=0.75)
        axis.set_ylabel(title, fontweight="bold")
        axis.set_title(f"{title} by Class", fontweight="bold")
        axis.set_ylim([0, 1])
        axis.tick_params(axis="x", rotation=45)
        axis.grid(True, alpha=0.3, axis="y")

    figure.tight_layout()
    figure.savefig(output_dir / "metrics_by_class.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataframe = load_data(args.max_rows)
    features, target, label_encoders, feature_names = prepare_features(dataframe)
    train_features, test_features, train_target, test_target = split_data(features, target)
    model = train_random_forest(train_features, train_target, feature_names)
    predictions = model.predict(test_features)
    metrics = evaluate_predictions(test_target, predictions, label_encoders)

    print("Saving CSV baseline plots...")
    plot_learning_curve(model, train_features, train_target, args.output_dir)
    plot_confusion_matrix(metrics, args.output_dir)
    plot_class_metrics(metrics, args.output_dir)

    print("\nCSV baseline completed.")
    print(f"Output folder: {args.output_dir}")


if __name__ == "__main__":
    main()
