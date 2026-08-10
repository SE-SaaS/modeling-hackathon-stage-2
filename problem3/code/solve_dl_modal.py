import modal
import os
import io
import pandas as pd

app = modal.App("problem3-dl-finetune")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "pandas",
        "scikit-learn",
        "accelerate"
    )
)

@app.function(gpu="A10G", image=image, timeout=3600)
def train_and_predict(train_csv_bytes: bytes, test_csv_bytes: bytes):
    import logging
    import pandas as pd
    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score
    from datasets import Dataset

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logging.info("Decoding CSVs...")
    train_df = pd.read_csv(io.BytesIO(train_csv_bytes))
    test_df = pd.read_csv(io.BytesIO(test_csv_bytes))
    
    train_df['text'] = train_df['text'].fillna('')
    test_df['text'] = test_df['text'].fillna('')
    
    model_name = "CAMeL-Lab/bert-base-arabic-camelbert-da"
    logging.info(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # We want to use early stopping, so we'll evaluate on a subset.
    # We use a standard random split for Deep Learning to learn general robust representations.
    train_split, val_split = train_test_split(train_df, test_size=0.1, stratify=train_df['dialect'], random_state=42)
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=64)
    
    logging.info("Tokenizing datasets...")
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
    
    logging.info("Loading pre-trained model weights...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {"macro_f1": f1_score(labels, predictions, average='macro')}
        
    training_args = TrainingArguments(
        output_dir='/tmp/results',
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        warmup_steps=100,
        weight_decay=0.01,
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
    
    # Return predictions as bytes of CSV
    logging.info("Encoding predictions...")
    out_df = pd.DataFrame({'id': test_df['id'], 'dialect': preds})
    csv_buffer = io.StringIO()
    out_df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue().encode('utf-8')

@app.local_entrypoint()
def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, 'datasets', 'train.csv')
    test_path = os.path.join(base_dir, 'datasets', 'test.csv')
    output_path = os.path.join(base_dir, 'output', 'submission_dl_modal.csv')
    
    print(f"Reading local datasets:\n - {train_path}\n - {test_path}")
    with open(train_path, 'rb') as f:
        train_bytes = f.read()
    with open(test_path, 'rb') as f:
        test_bytes = f.read()
        
    print("Uploading data and starting remote fine-tuning on Modal GPU...")
    print("This will take about 5-10 minutes. Check your Modal dashboard for real-time logs!")
    
    result_bytes = train_and_predict.remote(train_bytes, test_bytes)
    
    print(f"Success! Saving predictions to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(result_bytes)
    print("Done!")
