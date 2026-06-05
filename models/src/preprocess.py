"""Text preprocessing and BERT embedding utilities for SMS spam detection."""

from __future__ import annotations

from contextlib import nullcontext, redirect_stderr, redirect_stdout
import logging
import os
import re
import warnings

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, logging as transformers_logging


BERT_MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
transformers_logging.set_verbosity_error()
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")


def preprocess_text(text: object) -> str:
    """Normalize one SMS message before BERT tokenization."""
    if text is None:
        return ""

    cleaned = str(text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def preprocess_series(messages):
    """Apply SMS preprocessing to a pandas Series-like object."""
    return messages.fillna("").apply(preprocess_text)


def get_torch_device() -> torch.device:
    """Return CUDA device when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_bert_encoder(
    model_name: str = BERT_MODEL_NAME,
    device: torch.device | None = None,
    quiet: bool = True,
):
    """Load uncased BERT tokenizer and model for feature extraction."""
    selected_device = device or get_torch_device()
    if quiet:
        devnull = open(os.devnull, "w", encoding="utf-8")
        stdout_context = redirect_stdout(devnull)
        stderr_context = redirect_stderr(devnull)
    else:
        devnull = None
        stdout_context = nullcontext()
        stderr_context = nullcontext()

    try:
        with stdout_context, stderr_context:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
    finally:
        if devnull is not None:
            devnull.close()

    model.to(selected_device)
    model.eval()
    return tokenizer, model, selected_device


def mean_pool_bert(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Create one sentence embedding by averaging non-padding token vectors."""
    input_mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed_embeddings = torch.sum(last_hidden_state * input_mask, dim=1)
    token_counts = torch.clamp(input_mask.sum(dim=1), min=1e-9)
    return summed_embeddings / token_counts


def encode_bert_messages(
    messages,
    tokenizer,
    model,
    device: torch.device,
    batch_size: int = 16,
    max_length: int = MAX_LENGTH,
) -> np.ndarray:
    """Convert SMS messages to dense BERT embeddings for KNN."""
    cleaned_messages = [preprocess_text(message) for message in messages]
    embeddings = []

    with torch.no_grad():
        for start_index in range(0, len(cleaned_messages), batch_size):
            batch = cleaned_messages[start_index : start_index + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded)
            pooled = mean_pool_bert(outputs.last_hidden_state, encoded["attention_mask"])
            embeddings.append(pooled.cpu().numpy())

    if not embeddings:
        return np.empty((0, model.config.hidden_size))

    return np.vstack(embeddings)
