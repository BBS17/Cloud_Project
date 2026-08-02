"""
Baseline Model Evaluation Script
Tests current model on sample data to establish baseline accuracy
This helps us measure improvement after retraining with v2 script
"""

import pandas as pd
import numpy as np
import argparse
import json
import os
from pathlib import Path
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
DEFAULT_DATA_DIR = Path(os.getenv("ISOT_DATA_DIR", Path(__file__).parent / "data"))
ISOT_FAKE_PATH = Path(os.getenv("ISOT_FAKE_PATH", DEFAULT_DATA_DIR / "Fake.csv"))
ISOT_TRUE_PATH = Path(os.getenv("ISOT_TRUE_PATH", DEFAULT_DATA_DIR / "True.csv"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", Path(__file__).parent / "misinformation_model"))


def load_current_model(model_path=MODEL_PATH):
    """Load current production model"""
    logger.info(f"Loading model from {model_path}")

    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)

    # Create pipeline
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device
    )

    return classifier


def load_test_data(sample_size=500, fake_path=ISOT_FAKE_PATH, true_path=ISOT_TRUE_PATH):
    """Load sample from ISOT dataset for testing"""
    logger.info(f"Loading test data (sample_size={sample_size})...")

    if not fake_path.is_file() or not true_path.is_file():
        raise FileNotFoundError("ISOT CSVs not found; pass --fake-csv and --true-csv.")
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    # Combine title + text like v2 does
    fake_df['text'] = (fake_df['title'].fillna('') + " " + fake_df['text'].fillna('')).str.strip()
    true_df['text'] = (true_df['title'].fillna('') + " " + true_df['text'].fillna('')).str.strip()

    # Create labels
    fake_df['label'] = 0
    true_df['label'] = 1

    # Use the exact held-out partition created by train_model_v2. This avoids
    # evaluating v2 on articles it saw during training.
    all_data = pd.concat(
        [fake_df[['text', 'label']], true_df[['text', 'label']]],
        ignore_index=True,
    )
    _, holdout = train_test_split(
        all_data, test_size=0.30, random_state=42, stratify=all_data['label']
    )
    _, test_data = train_test_split(
        holdout, test_size=0.50, random_state=42, stratify=holdout['label']
    )
    if sample_size and sample_size < len(test_data):
        test_data, _ = train_test_split(
            test_data,
            train_size=sample_size,
            random_state=42,
            stratify=test_data['label'],
        )
    test_data = test_data.sample(frac=1, random_state=42).reset_index(drop=True)

    logger.info(f"Test data loaded: {len(test_data)} samples")
    logger.info(f"  - Fake (0): {len(test_data[test_data['label'] == 0])}")
    logger.info(f"  - True (1): {len(test_data[test_data['label'] == 1])}")

    return test_data


def predict_batch(texts, classifier, batch_size=32):
    """Predict labels for texts in batches"""
    predictions = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        results = classifier(batch, truncation=True, max_length=256)

        for result in results:
            # Convert label string to int: "LABEL_0" -> 0, "LABEL_1" -> 1
            label = int(result['label'].split('_')[1])
            score = result['score']
            predictions.append({'label': label, 'score': score})

    return predictions


def evaluate(model_path=MODEL_PATH, fake_path=ISOT_FAKE_PATH,
             true_path=ISOT_TRUE_PATH, sample_size=500, report_path=None):
    """Run evaluation"""
    logger.info("="*60)
    logger.info("BASELINE MODEL EVALUATION")
    logger.info("="*60)

    classifier = load_current_model(model_path)
    test_data = load_test_data(sample_size, fake_path, true_path)

    # Predict
    logger.info("Running predictions...")
    predictions = predict_batch(test_data['text'].tolist(), classifier)

    pred_labels = [p['label'] for p in predictions]
    pred_scores = [p['score'] for p in predictions]
    true_labels = test_data['label'].tolist()

    # Compute metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)

    cm = confusion_matrix(true_labels, pred_labels)
    tn, fp, fn, tp = cm.ravel()

    logger.info("\n" + "="*60)
    logger.info("RESULTS")
    logger.info("="*60)
    logger.info(f"Accuracy:  {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1 Score:  {f1:.4f}")
    logger.info("\nConfusion Matrix:")
    logger.info(f"  True Negatives (Correctly classified fake):   {tn}")
    logger.info(f"  False Positives (Fake marked as true):        {fp}")
    logger.info(f"  False Negatives (True marked as fake):        {fn}")
    logger.info(f"  True Positives (Correctly classified true):   {tp}")
    logger.info(f"\nAverage Confidence: {np.mean(pred_scores):.4f}")
    logger.info(f"Min Confidence: {np.min(pred_scores):.4f}")
    logger.info(f"Max Confidence: {np.max(pred_scores):.4f}")

    # Find misclassifications
    misclassified = []
    for i, (true_label, pred_label, score) in enumerate(zip(true_labels, pred_labels, pred_scores)):
        if true_label != pred_label:
            text = test_data['text'].iloc[i]
            misclassified.append({
                'text': text[:100],
                'true_label': true_label,
                'pred_label': pred_label,
                'score': score
            })

    logger.info(f"\nSample Misclassifications ({len(misclassified)} total):")
    for i, m in enumerate(misclassified[:5]):
        true_label_str = "True" if m['true_label'] == 1 else "Fake"
        pred_label_str = "True" if m['pred_label'] == 1 else "Fake"
        logger.info(f"  {i+1}. Actual: {true_label_str}, Predicted: {pred_label_str} ({m['score']:.2%})")
        logger.info(f"     Text: {m['text']}...")

    logger.info("="*60)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'test_size': len(test_data),
        'misclassified_count': len(misclassified)
    }
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        logger.info("Saved metrics to %s", report_path)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a fact-checker checkpoint")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--fake-csv", type=Path, default=ISOT_FAKE_PATH)
    parser.add_argument("--true-csv", type=Path, default=ISOT_TRUE_PATH)
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--report", type=Path, default=Path(__file__).parent / "baseline_metrics.json")
    args = parser.parse_args()
    evaluate(args.model_path, args.fake_csv, args.true_csv, args.sample_size, args.report)
