import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import os

def load_data():
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    return train_df, test_df

def get_word_counts(texts):
    return np.array([len(str(t).split()) for t in texts])

def run_baseline():
    print("Loading data...")
    train_df, test_df = load_data()
    
    # Fill any potential NaNs
    train_df['text'] = train_df['text'].fillna('')
    test_df['text'] = test_df['text'].fillna('')
    
    X_train = train_df['text'].values
    y_train = train_df['dialect'].values
    X_test = test_df['text'].values
    
    print("Extracting features using TF-IDF (Word and Char N-grams)...")
    # Feature union to combine word and character n-grams
    word_vectorizer = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 3),
        max_features=40000,
        sublinear_tf=True
    )
    char_vectorizer = TfidfVectorizer(
        analyzer='char',
        ngram_range=(2, 5),
        max_features=80000,
        sublinear_tf=True
    )
    
    vectorizer = FeatureUnion([
        ("word", word_vectorizer),
        ("char", char_vectorizer)
    ])
    
    # Custom Validation Logic: Focus on short sentences (<= 8 words)
    print("Running Custom Cross-Validation focusing on short sentences...")
    word_counts = get_word_counts(X_train)
    short_sentence_mask = word_counts <= 8
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    val_f1_scores = []
    short_val_f1_scores = []
    
    # We will fit transform on the entire train data for faster CV (slight data leakage for TF-IDF but acceptable for quick baseline)
    X_train_vec = vectorizer.fit_transform(X_train)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_vec, y_train)):
        X_tr, y_tr = X_train_vec[train_idx], y_train[train_idx]
        X_va, y_va = X_train_vec[val_idx], y_train[val_idx]
        
        clf = LinearSVC(C=0.5, class_weight='balanced', random_state=42, dual=False)
        clf.fit(X_tr, y_tr)
        
        preds = clf.predict(X_va)
        
        # Overall F1
        score = f1_score(y_va, preds, average='macro')
        val_f1_scores.append(score)
        
        # Short sentences F1
        val_short_mask = short_sentence_mask[val_idx]
        if val_short_mask.sum() > 0:
            short_score = f1_score(y_va[val_short_mask], preds[val_short_mask], average='macro')
            short_val_f1_scores.append(short_score)
        
        print(f"Fold {fold+1}: Overall F1 = {score:.4f}, Short-Sentence F1 = {short_score:.4f}")
        
    print(f"Mean Overall CV F1: {np.mean(val_f1_scores):.4f}")
    print(f"Mean Short-Sentence CV F1: {np.mean(short_val_f1_scores):.4f}")
    
    print("Training final model on full dataset...")
    clf_final = LinearSVC(C=0.5, class_weight='balanced', random_state=42, dual=False)
    clf_final.fit(X_train_vec, y_train)
    
    print("Predicting on test set...")
    X_test_vec = vectorizer.transform(X_test)
    test_preds = clf_final.predict(X_test_vec)
    
    print("Generating submission...")
    sub = pd.read_csv('sample_submission.csv')
    sub['dialect'] = test_preds
    sub.to_csv('submission_baseline.csv', index=False)
    print("Baseline submission saved to 'submission_baseline.csv'.")

if __name__ == "__main__":
    run_baseline()
