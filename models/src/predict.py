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

def load_artifacts(model_type: str = "knn"):
    if model_type == "knn":
        model = joblib.load(MODEL_PATH)
        config = json.loads(BERT_CONFIG_PATH.read_text(encoding="utf-8"))
        if config.get("pipeline") != "bert_knn":
            raise ValueError("Saved model is not a BERT + KNN artifact. Run training again.")
        config.setdefault("model_name", BERT_MODEL_NAME)
        config.setdefault("max_length", MAX_LENGTH)
        return model, config, "bert"
    elif model_type == "nn":
        model = joblib.load(PROJECT_ROOT / "model_TF-IDF_NeuralNetwork(MLP)_SMS_Spam.pkl")
        return model, None, "pipeline"
    elif model_type == "dt":
        model = joblib.load(PROJECT_ROOT / "decision_tree_model.pkl")
        vectorizer = joblib.load(PROJECT_ROOT / "vectorizer.pkl")
        return model, vectorizer, "tfidf_standalone"
    elif model_type == "svm":
        model = joblib.load(PROJECT_ROOT / "SVM_model.pkl")
        vectorizer = joblib.load(PROJECT_ROOT / "vectorizer.pkl")
        return model, vectorizer, "tfidf_standalone"
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def predict_sms(message: str, model_type: str = "knn") -> str:
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Message must be a non-empty string.")

    model, extra, m_type = load_artifacts(model_type)
    
    if m_type == "bert":
        tokenizer, bert_model, device = load_bert_encoder(extra["model_name"])
        features = encode_bert_messages([message], tokenizer, bert_model, device, batch_size=1, max_length=int(extra["max_length"]))
        prediction = model.predict(features)[0]
    elif m_type == "pipeline":
        prediction = model.predict([message])[0]
    elif m_type == "tfidf_standalone":
        vectorizer = extra
        features = vectorizer.transform([message])
        prediction = model.predict(features)[0]

    try:
        pred_int = int(prediction)
        return "Spam" if pred_int == 1 else "Ham"
    except ValueError:
        return "Spam" if str(prediction).lower() == "spam" else "Ham"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict SMS spam/ham label.")
    parser.add_argument("--model", type=str, default="knn", choices=["knn", "dt", "nn", "svm"], help="Model type to use")
    parser.add_argument("message", nargs="*", help="SMS text to classify.")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    message = " ".join(args.message).strip()

    if not message:
        try:
            message = input().strip()
        except EOFError as exc:
            raise SystemExit("Prediction failed: no message provided.") from exc

    try:
        print(predict_sms(message, args.model))
    except Exception as exc:
        raise SystemExit(f"Prediction failed: {exc}") from exc

if __name__ == "__main__":
    main()
