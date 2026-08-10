import os
import logging
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from datasets import Dataset

def get_word_counts(texts):
    return np.array([len(str(t).split()) for t in texts])

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"macro_f1": f1_score(labels, predictions, average='macro')}

def run_dl():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"CUDA Available: {torch.cuda.is_available()}")
    model_name = "CAMeL-Lab/bert-base-arabic-camelbert-da"
    logging.info(f"Using model: {model_name}")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, 'datasets', 'train.csv')
    test_path = os.path.join(base_dir, 'datasets', 'test.csv')
    output_path = os.path.join(base_dir, 'output', 'submission_dl.csv')

    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        return
    
    train_df['text'] = train_df['text'].fillna('')
    test_df['text'] = test_df['text'].fillna('')
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Validation logic: 
    # To truly simulate the private leaderboard, we could sample a validation set with median len = 8.
    # For simplicity and speed in fine-tuning, let's do a 90/10 split but favor short sentences for validation.
    
    word_counts = get_word_counts(train_df['text'].values)
    short_sentence_mask = word_counts <= 12
    
    # We will pick 10% of the short sentences as our validation set
    short_indices = np.where(short_sentence_mask)[0]
    np.random.seed(42)
    val_indices = np.random.choice(short_indices, size=int(0.1 * len(train_df)), replace=False)
    train_indices = np.setdiff1d(np.arange(len(train_df)), val_indices)
    
    train_split = train_df.iloc[train_indices].copy()
    val_split = train_df.iloc[val_indices].copy()
    
    logging.info(f"Train size: {len(train_split)}, Val size (short sentences): {len(val_split)}")

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=64)
    
    train_dataset = Dataset.from_pandas(train_split[['text', 'dialect']])
    val_dataset = Dataset.from_pandas(val_split[['text', 'dialect']])
    test_dataset = Dataset.from_pandas(test_df[['text']])
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    train_dataset = train_dataset.rename_column("dialect", "labels")
    val_dataset = val_dataset.rename_column("dialect", "labels")
    
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)
    
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    logging.info("Training model...")
    trainer.train()
    
    logging.info("Evaluating on validation set...")
    val_results = trainer.evaluate()
    logging.info(f"Validation Results: {val_results}")
    
    logging.info("Predicting on test set...")
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    
    sub = pd.DataFrame({'id': test_df['id'], 'dialect': preds})
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sub.to_csv(output_path, index=False)
    logging.info(f"Deep learning submission saved to '{output_path}'.")

if __name__ == "__main__":
    run_dl()
