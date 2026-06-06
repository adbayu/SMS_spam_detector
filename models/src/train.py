"""Train and evaluate BERT + KNN SMS spam detector."""

from __future__ import annotations

import argparse
import json
from io import StringIO
import os
from pathlib import Path
import sys

import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.preprocess import (
    BERT_MODEL_NAME,
    MAX_LENGTH,
    encode_bert_messages,
    load_bert_encoder,
    preprocess_series,
)


DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "spam.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
RANDOM_STATE = 42
BERT_CONFIG_PATH = MODELS_DIR / "bert_config.json"


def load_dataset(data_path: Path) -> pd.DataFrame:
    """Load Kaggle SMS Spam Collection and normalize columns."""
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path, encoding="latin-1")

    if {"label", "text"}.issubset(df.columns):
        df = df[["label", "text"]].copy()
    elif {"v1", "v2"}.issubset(df.columns):
        df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "text"})
    else:
        raise ValueError("Dataset must contain label/text or v1/v2 columns.")

    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["label"].isin(["ham", "spam"])].copy()
    return df


def sample_dataset(df: pd.DataFrame, sample_size: int | None) -> pd.DataFrame:
    """Optionally sample a stratified subset for quick smoke training."""
    if sample_size is None or sample_size >= len(df):
        return df
    if sample_size < 10:
        raise ValueError("Sample size must be at least 10.")

    sampled_groups = []
    for _, group in df.groupby("label"):
        group_sample_size = max(1, round(sample_size * len(group) / len(df)))
        sampled_groups.append(
            group.sample(
                min(group_sample_size, len(group)),
                random_state=RANDOM_STATE,
            )
        )

    return pd.concat(sampled_groups).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def build_data_understanding(df: pd.DataFrame) -> str:
    """Create dataset inspection report text."""
    buffer = StringIO()
    df.info(buf=buffer)

    missing_values = df.isna().sum()
    duplicate_count = int(df.duplicated().sum())
    class_counts = df["label"].value_counts()
    class_percent = (df["label"].value_counts(normalize=True) * 100).round(2)

    return (
        "DATA UNDERSTANDING\n"
        "==================\n\n"
        f"Dataset shape: {df.shape}\n\n"
        "Dataset info:\n"
        f"{buffer.getvalue()}\n"
        "Missing values:\n"
        f"{missing_values.to_string()}\n\n"
        f"Duplicate rows: {duplicate_count}\n\n"
        "Class counts:\n"
        f"{class_counts.to_string()}\n\n"
        "Class percentage:\n"
        f"{class_percent.to_string()}\n\n"
    )


def prepare_features(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Clean text and encode labels: spam=1, ham=0."""
    x = preprocess_series(df["text"])
    y = df["label"].map({"ham": 0, "spam": 1})
    return x, y

def oversample_training_data(
    x_train: pd.Series,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.Series, pd.Series]:
    """Randomly oversample minority class in train data only."""
    train_df = pd.DataFrame({"text": x_train.reset_index(drop=True), "label": y_train.reset_index(drop=True)})
    class_counts = train_df["label"].value_counts()

    if len(class_counts) < 2 or class_counts.min() == class_counts.max():
        return x_train.reset_index(drop=True), y_train.reset_index(drop=True)

    target_count = int(class_counts.max())
    balanced_parts = []

    for label, group in train_df.groupby("label"):
        balanced_parts.append(
            group.sample(
                n=target_count,
                replace=len(group) < target_count,
                random_state=random_state + int(label),
            )
        )

    balanced_df = pd.concat(balanced_parts).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return balanced_df["text"], balanced_df["label"]

def format_label_counts(y: pd.Series) -> dict[str, int]:
    """Format encoded class counts with readable labels."""
    counts = y.value_counts().sort_index()
    return {
        "ham": int(counts.get(0, 0)),
        "spam": int(counts.get(1, 0)),
    }


def train_grid_search(x_train_embeddings, y_train: pd.Series, cv: int = 5) -> GridSearchCV:
    """Train KNN with BERT embeddings using GridSearchCV."""
    if y_train.value_counts().min() < cv:
        cv = int(y_train.value_counts().min())

    param_grid = {
        "n_neighbors": [3, 5, 7, 9, 11, 13, 15],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "cosine"],
    }

    grid_search = GridSearchCV(
        estimator=KNeighborsClassifier(),
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=1,
        verbose=1,
    )
    grid_search.fit(x_train_embeddings, y_train)
    return grid_search


def evaluate_model(model: KNeighborsClassifier, x_test_embeddings, y_test: pd.Series) -> dict:
    """Evaluate fitted model on test data."""
    y_pred = model.predict(x_test_embeddings)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=["Ham", "Spam"],
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }


def save_confusion_matrix(cm, output_path: Path, oversampling_report: str) -> None:
    """Save confusion matrix heatmap."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Ham", "Spam"],
        yticklabels=["Ham", "Spam"],
    )
    plt.title("SMS Spam Detection - Confusion Matrix")
    plt.suptitle(oversampling_report.replace("\n", " | "), fontsize=9, y=0.98)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_artifacts(
    model: KNeighborsClassifier,
    model_name: str,
    max_length: int,
) -> None:
    """Save final KNN model and BERT encoder configuration."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "best_knn_model.pkl")
    BERT_CONFIG_PATH.write_text(
        json.dumps(
            {
                "pipeline": "bert_knn",
                "model_name": model_name,
                "max_length": max_length,
                "embedding": "mean_pool_last_hidden_state",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def build_evaluation_report(
    grid_search: GridSearchCV,
    metrics: dict,
    data_report: str,
    oversampling_report: str,
) -> str:
    """Build final report text for reports/evaluation.txt."""
    best_params = grid_search.best_params_
    result_text = (
        "MODEL EVALUATION\n"
        "================\n\n"
        "Feature extractor: bert-base-uncased embeddings\n"
        "Classifier: K-Nearest Neighbors\n\n"
        "OVERSAMPLING SUMMARY\n"
        "--------------------\n"
        f"{oversampling_report}\n\n"
        f"Best CV F1 score: {grid_search.best_score_:.4f}\n"
        f"Best parameters: {best_params}\n\n"
        f"Accuracy: {metrics['accuracy']:.4f}\n"
        f"Precision: {metrics['precision']:.4f}\n"
        f"Recall: {metrics['recall']:.4f}\n"
        f"F1 Score: {metrics['f1']:.4f}\n\n"
        "Classification Report:\n"
        f"{metrics['classification_report']}\n"
        "Confusion Matrix [[TN, FP], [FN, TP]]:\n"
        f"{metrics['confusion_matrix'].tolist()}\n\n"
        "Result Explanation:\n"
        "- Accuracy shows overall correct predictions.\n"
        "- Precision shows how many predicted spam messages are truly spam.\n"
        "- Recall shows how many true spam messages are detected.\n"
        "- F1 balances precision and recall, important for imbalanced data.\n"
        "- BERT converts each SMS into a dense semantic embedding.\n"
        "- KNN was selected by highest cross-validation F1 score on BERT embeddings.\n"
    )
    return f"{data_report}\n{result_text}"


def run_training(
    data_path: Path = DEFAULT_DATA_PATH,
    model_name: str = BERT_MODEL_NAME,
    batch_size: int = 16,
    max_length: int = MAX_LENGTH,
    sample_size: int | None = None,
    oversample: bool = True,
) -> dict:
    """Run full training pipeline and save artifacts/reports."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = sample_dataset(load_dataset(data_path), sample_size)
    data_report = build_data_understanding(df)
    x, y = prepare_features(df)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    before_counts = format_label_counts(y_train)
    if oversample:
        x_train, y_train = oversample_training_data(x_train, y_train)
    after_counts = format_label_counts(y_train)
    oversampling_report = (
        "Oversampling: random minority oversampling on train set only\n"
        f"Train class counts before: {before_counts}\n"
        f"Train class counts after: {after_counts}"
    )
    print(oversampling_report, flush=True)

    print(f"Loading BERT encoder: {model_name}", flush=True)
    tokenizer, bert_model, device = load_bert_encoder(model_name)
    print(f"Encoding train messages on {device}...", flush=True)
    x_train_embeddings = encode_bert_messages(
        x_train,
        tokenizer,
        bert_model,
        device,
        batch_size=batch_size,
        max_length=max_length,
    )
    print("Encoding test messages...", flush=True)
    x_test_embeddings = encode_bert_messages(
        x_test,
        tokenizer,
        bert_model,
        device,
        batch_size=batch_size,
        max_length=max_length,
    )

    print("Running KNN GridSearchCV...", flush=True)
    grid_search = train_grid_search(x_train_embeddings, y_train)
    best_model = grid_search.best_estimator_
    metrics = evaluate_model(best_model, x_test_embeddings, y_test)

    save_artifacts(best_model, model_name, max_length)
    save_confusion_matrix(
        metrics["confusion_matrix"],
        REPORTS_DIR / "confusion_matrix.png",
        oversampling_report,
    )

    evaluation_text = build_evaluation_report(grid_search, metrics, data_report, oversampling_report)
    (REPORTS_DIR / "evaluation.txt").write_text(evaluation_text, encoding="utf-8")

    return {
        "best_params": grid_search.best_params_,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Train KNN SMS spam detector.")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to SMS spam CSV.",
    )
    parser.add_argument(
        "--bert-model",
        default=BERT_MODEL_NAME,
        help="Hugging Face BERT model name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="BERT embedding batch size.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=MAX_LENGTH,
        help="Maximum BERT token length.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional stratified sample size for quick smoke training.",
    )
    parser.add_argument(
        "--no-oversampling",
        action="store_true",
        help="Disable random oversampling on minority spam class in training set.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    try:
        summary = run_training(
            data_path=args.data,
            model_name=args.bert_model,
            batch_size=args.batch_size,
            max_length=args.max_length,
            sample_size=args.sample_size,
            oversample=not args.no_oversampling,
        )
    except Exception as exc:
        raise SystemExit(f"Training failed: {exc}") from exc

    print("Training complete.")
    print(f"Best params: {summary['best_params']}")
    print(f"Accuracy: {summary['accuracy']:.4f}")
    print(f"Precision: {summary['precision']:.4f}")
    print(f"Recall: {summary['recall']:.4f}")
    print(f"F1: {summary['f1']:.4f}")


if __name__ == "__main__":
    main()
