"""Standalone prediction script for SMS spam detection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.preprocess import BERT_MODEL_NAME, MAX_LENGTH, encode_bert_messages, load_bert_encoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "best_knn_model.pkl"
BERT_CONFIG_PATH = PROJECT_ROOT / "bert_config.json"

def load_artifacts(model_path: Path = MODEL_PATH, config_path: Path = BERT_CONFIG_PATH):
    """Load saved KNN model and BERT configuration."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not config_path.exists():
        raise FileNotFoundError(
            f"BERT config not found: {config_path}. Run `python src/train.py` first."
        )

    model = joblib.load(model_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("pipeline") != "bert_knn":
        raise ValueError("Saved model is not a BERT + KNN artifact. Run training again.")
    config.setdefault("model_name", BERT_MODEL_NAME)
    config.setdefault("max_length", MAX_LENGTH)
    return model, config

def predict_sms(message: str) -> str:
    """Predict SMS label. Returns 'Spam' or 'Ham'."""
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Message must be a non-empty string.")

    model, config = load_artifacts()
    tokenizer, bert_model, device = load_bert_encoder(config["model_name"])
    features = encode_bert_messages(
        [message],
        tokenizer,
        bert_model,
        device,
        batch_size=1,
        max_length=int(config["max_length"]),
    )
    prediction = model.predict(features)[0]
    return "Spam" if int(prediction) == 1 else "Ham"

def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Predict SMS spam/ham label.")
    parser.add_argument("message", nargs="*", help="SMS text to classify.")
    return parser.parse_args()

def main() -> None:
    """CLI entrypoint for PHP-friendly integration."""
    args = parse_args()
    message = " ".join(args.message).strip()

    if not message:
        try:
            message = input().strip()
        except EOFError as exc:
            raise SystemExit("Prediction failed: no message provided.") from exc

    try:
        print(predict_sms(message))
    except Exception as exc:
        raise SystemExit(f"Prediction failed: {exc}") from exc

if __name__ == "__main__":
    main()
