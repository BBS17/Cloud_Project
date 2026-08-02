import logging
import os
from pathlib import Path

from app.model_labels import display_label

logger = logging.getLogger(__name__)

MODEL_PATH = Path(os.getenv("MODEL_PATH", Path(__file__).parent / "misinformation_model"))

# Initialised to None; populated by load_model() at startup
tokenizer = None
model = None

# The ISOT training scripts use 0 = fake and 1 = true.  Keep this fallback for
# older checkpoints that were saved without id2label metadata.


def is_model_loaded() -> bool:
    return model is not None


def load_model() -> None:
    """Load the tokenizer and model into memory. Safe to call multiple times."""
    global tokenizer, model
    if model is not None:
        return

    from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

    try:
        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
        model.eval()
        logger.info("Model loaded from %s", MODEL_PATH)
    except Exception as exc:
        logger.critical("Failed to load model from %s: %s", MODEL_PATH, exc)
        raise


def predict_text(text: str) -> dict:
    import torch

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    confidence = round(probabilities[0][predicted_class].item() * 100, 2)

    return {
        "label": display_label(predicted_class, model.config),
        "confidence": confidence,
    }
