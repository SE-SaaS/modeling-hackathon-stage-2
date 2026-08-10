import modal
import os
import io
import pandas as pd

app = modal.App("problem3-dl-kfold")

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

@app.function(gpu="A10G", image=image, timeout=7200) # 2 hours timeout
def train_and_predict_kfold(train_csv_bytes: bytes, test_csv_bytes: bytes):
    import logging
    import pandas as pd
    import numpy as np
    import torch
    import gc
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
    from sklearn.model_selection import StratifiedKFold
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
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=64)

    logging.info("Preparing test dataset...")
    test_dataset = Dataset.from_pandas(test_df[['text']])
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Store logits from all 5 models
    # Shape will be (5, num_test_samples, num_classes)
    all_logits = np.zeros((5, len(test_df), 4))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['dialect'])):
        logging.info(f"=== Starting Fold {fold + 1} / 5 ===")
        
        train_split = train_df.iloc[train_idx]
        val_split = train_df.iloc[val_idx]
        
        train_dataset = Dataset.from_pandas(train_split[['text', 'dialect']])
        val_dataset = Dataset.from_pandas(val_split[['text', 'dialect']])
        
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)
        
        train_dataset = train_dataset.rename_column("dialect", "labels")
        val_dataset = val_dataset.rename_column("dialect", "labels")
        
        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)
        
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return {"macro_f1": f1_score(labels, predictions, average='macro')}
            
        training_args = TrainingArguments(
            output_dir=f'/tmp/results_fold_{fold}',
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
        
        logging.info(f"Training model for Fold {fold+1}...")
        trainer.train()
        
        val_results = trainer.evaluate()
        logging.info(f"Fold {fold+1} Validation Results: {val_results}")
        
        logging.info(f"Predicting on test set with Fold {fold+1} model...")
        predictions = trainer.predict(test_dataset)
        
        # Save raw logits for this fold
        all_logits[fold] = predictions.predictions
        
        # Cleanup to free GPU memory
        del model
        del trainer
        torch.cuda.empty_cache()
        gc.collect()
        logging.info(f"=== Completed Fold {fold + 1} ===")

    logging.info("Ensembling logits from all 5 folds...")
    avg_logits = np.mean(all_logits, axis=0)
    final_preds = np.argmax(avg_logits, axis=-1)
    
    logging.info("Encoding predictions...")
    out_df = pd.DataFrame({'id': test_df['id'], 'dialect': final_preds})
    csv_buffer = io.StringIO()
    out_df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue().encode('utf-8')

@app.local_entrypoint()
def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, 'datasets', 'train.csv')
    test_path = os.path.join(base_dir, 'datasets', 'test.csv')
    output_path = os.path.join(base_dir, 'output', 'submission_dl_modal_kfold.csv')
    
    print(f"Reading local datasets...")
    with open(train_path, 'rb') as f:
        train_bytes = f.read()
    with open(test_path, 'rb') as f:
        test_bytes = f.read()
        
    print("Uploading data and starting 5-Fold Ensembling on Modal GPU...")
    print("This will take approximately 30-40 minutes. Check your Modal dashboard for real-time logs!")
    
    result_bytes = train_and_predict_kfold.remote(train_bytes, test_bytes)
    
    print(f"Success! Saving ensembled predictions to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(result_bytes)
    print("Done!")
