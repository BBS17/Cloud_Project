"""
Enhanced Fact-Checking Model Training Script
Improvements over v1:
- Uses full article text instead of just titles
- Combines ISOT dataset with support for additional datasets
- Increased max_length (256 tokens) for better context
- Early stopping to prevent overfitting
- Better hyperparameters (3 epochs, learning rate 1e-5)
- More detailed validation metrics
- Progress logging with validation scores
"""

import os
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Model configuration
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256  # Increased from 128 to capture more context
NUM_EPOCHS = 3    # Increased from 2 for better convergence
BATCH_SIZE = 8
LEARNING_RATE = 1e-5  # Reduced for stability
EARLY_STOP_PATIENCE = 2

# Paths
DEFAULT_DATA_DIR = Path(os.getenv("ISOT_DATA_DIR", Path(__file__).parent / "data"))
ISOT_FAKE_PATH = Path(os.getenv("ISOT_FAKE_PATH", DEFAULT_DATA_DIR / "Fake.csv"))
ISOT_TRUE_PATH = Path(os.getenv("ISOT_TRUE_PATH", DEFAULT_DATA_DIR / "True.csv"))
OUTPUT_DIR = Path(__file__).parent / "misinformation_model"
TRAINING_LOG_FILE = Path(__file__).parent / "training_log.txt"


def load_isot_dataset(fake_path=ISOT_FAKE_PATH, true_path=ISOT_TRUE_PATH):
    """Load ISOT Fake and Real News dataset"""
    logger.info("Loading ISOT dataset...")

    if not fake_path.is_file() or not true_path.is_file():
        raise FileNotFoundError(
            "ISOT CSVs not found. Pass --fake-csv/--true-csv or set "
            "ISOT_FAKE_PATH and ISOT_TRUE_PATH."
        )
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    logger.info(f"Loaded {len(fake_df)} fake articles and {len(true_df)} true articles")

    # Use full text (title + text combined for better context)
    fake_df['text'] = (fake_df['title'].fillna('') + " " + fake_df['text'].fillna('')) .str.strip()
    true_df['text'] = (true_df['title'].fillna('') + " " + true_df['text'].fillna('')).str.strip()

    # Create labels (0 = fake, 1 = true)
    fake_df['label'] = 0
    true_df['label'] = 1

    # Select relevant columns
    fake_df = fake_df[['text', 'label']].dropna(subset=['text'])
    true_df = true_df[['text', 'label']].dropna(subset=['text'])

    # Remove very short texts (minimum 10 words)
    fake_df = fake_df[fake_df['text'].str.split().str.len() >= 10]
    true_df = true_df[true_df['text'].str.split().str.len() >= 10]

    # Combine datasets
    data = pd.concat([fake_df, true_df], ignore_index=True)

    logger.info(f"Combined dataset: {len(data)} articles")
    logger.info(f"  - Fake (label=0): {len(data[data['label'] == 0])}")
    logger.info(f"  - True (label=1): {len(data[data['label'] == 1])}")

    return data


def prepare_datasets(data, test_size=0.15, val_size=0.15):
    """Split data into train, validation, and test sets"""
    logger.info("Preparing train/val/test splits...")

    # First split: train + temp (val + test)
    train_data, temp_data = train_test_split(
        data,
        test_size=(val_size + test_size),
        random_state=42,
        stratify=data['label']
    )

    # Second split: val and test from temp
    val_data, test_data = train_test_split(
        temp_data,
        test_size=test_size / (val_size + test_size),
        random_state=42,
        stratify=temp_data['label']
    )

    logger.info(f"Train set: {len(train_data)} samples")
    logger.info(f"Validation set: {len(val_data)} samples")
    logger.info(f"Test set: {len(test_data)} samples")

    return train_data.reset_index(drop=True), val_data.reset_index(drop=True), test_data.reset_index(drop=True)


def tokenize_function(examples, tokenizer):
    """Tokenize text data"""
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=MAX_LENGTH
    )


def compute_metrics(eval_pred):
    """Compute evaluation metrics"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def train_model(fake_path=ISOT_FAKE_PATH, true_path=ISOT_TRUE_PATH, output_dir=OUTPUT_DIR):
    """Main training function"""
    logger.info("=" * 60)
    logger.info("Starting Model Training (v2 - Enhanced)")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("=" * 60)

    # Load and prepare data
    data = load_isot_dataset(fake_path, true_path)
    train_data, val_data, test_data = prepare_datasets(data)

    # Load tokenizer
    logger.info(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    # Tokenize datasets
    logger.info("Tokenizing datasets...")
    train_dataset = Dataset.from_dict({
        'text': train_data['text'].tolist(),
        'label': train_data['label'].tolist()
    })
    val_dataset = Dataset.from_dict({
        'text': val_data['text'].tolist(),
        'label': val_data['label'].tolist()
    })
    test_dataset = Dataset.from_dict({
        'text': test_data['text'].tolist(),
        'label': test_data['label'].tolist()
    })

    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=['text']
    )
    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=['text']
    )
    test_dataset = test_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=['text']
    )

    # Load model
    logger.info(f"Loading model: {MODEL_NAME}")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "Misinformation", 1: "Truth"},
        label2id={"Misinformation": 0, "Truth": 1},
    )

    # Create temporary training directory
    temp_dir = Path(__file__).parent / ".training_tmp"
    temp_dir.mkdir(exist_ok=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(temp_dir),
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(),  # Use mixed precision if CUDA available
        seed=42
    )

    # Early stopping callback
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=EARLY_STOP_PATIENCE,
        early_stopping_threshold=0.0
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[early_stopping],
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer)
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)

    logger.info("=" * 60)
    logger.info("Test Set Results:")
    logger.info(f"  Accuracy:  {test_results['eval_accuracy']:.4f}")
    logger.info(f"  Precision: {test_results['eval_precision']:.4f}")
    logger.info(f"  Recall:    {test_results['eval_recall']:.4f}")
    logger.info(f"  F1 Score:  {test_results['eval_f1']:.4f}")
    logger.info("=" * 60)

    # Save model and tokenizer
    logger.info(f"Saving model to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training metrics
    with open(TRAINING_LOG_FILE, 'a') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Training Completed: {datetime.now().isoformat()}\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Max Length: {MAX_LENGTH}\n")
        f.write(f"Epochs: {NUM_EPOCHS}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Learning Rate: {LEARNING_RATE}\n")
        f.write("\nTest Results:\n")
        f.write(f"  Accuracy:  {test_results['eval_accuracy']:.4f}\n")
        f.write(f"  Precision: {test_results['eval_precision']:.4f}\n")
        f.write(f"  Recall:    {test_results['eval_recall']:.4f}\n")
        f.write(f"  F1 Score:  {test_results['eval_f1']:.4f}\n")
        f.write(f"{'='*60}\n")

    # Cleanup temporary directory
    import shutil
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    logger.info("Training complete!")
    return test_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the v2 fact-checker model")
    parser.add_argument("--fake-csv", type=Path, default=ISOT_FAKE_PATH)
    parser.add_argument("--true-csv", type=Path, default=ISOT_TRUE_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    train_model(args.fake_csv, args.true_csv, args.output_dir)
