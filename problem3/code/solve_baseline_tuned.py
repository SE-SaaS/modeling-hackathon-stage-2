import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, 'datasets', 'train.csv')
    test_path = os.path.join(base_dir, 'datasets', 'test.csv')
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def get_word_counts(texts):
    return np.array([len(str(t).split()) for t in texts])

def run_tuned_baseline():
    print("Loading data...")
    train_df, test_df = load_data()
    
    train_df['text'] = train_df['text'].fillna('')
    test_df['text'] = test_df['text'].fillna('')
    
    X_train = train_df['text'].values
    y_train = train_df['dialect'].values
    X_test = test_df['text'].values
    
    print("Extracting features using TF-IDF (Word, Char, and Char-WB N-grams)...")
    word_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), max_features=80000, sublinear_tf=True)
    char_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 5), max_features=150000, sublinear_tf=True)
    char_wb_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), max_features=150000, sublinear_tf=True)
    
    vectorizer = FeatureUnion([
        ("word", word_vectorizer),
        ("char", char_vectorizer),
        ("char_wb", char_wb_vectorizer)
    ])
    
    word_counts = get_word_counts(X_train)
    short_sentence_mask = word_counts <= 8
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("Fitting TF-IDF on full training data (this may take a minute)...")
    X_train_vec = vectorizer.fit_transform(X_train)
    
    C_values = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0]
    best_c = None
    best_short_f1 = 0
    
    print(f"Starting Hyperparameter Search for C in {C_values}...")
    
    for c in C_values:
        val_f1_scores = []
        short_val_f1_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_vec, y_train)):
            X_tr, y_tr = X_train_vec[train_idx], y_train[train_idx]
            X_va, y_va = X_train_vec[val_idx], y_train[val_idx]
            
            clf = LinearSVC(C=c, class_weight='balanced', random_state=42, dual=False)
            clf.fit(X_tr, y_tr)
            
            preds = clf.predict(X_va)
            val_f1_scores.append(f1_score(y_va, preds, average='macro'))
            
            val_short_mask = short_sentence_mask[val_idx]
            if val_short_mask.sum() > 0:
                short_val_f1_scores.append(f1_score(y_va[val_short_mask], preds[val_short_mask], average='macro'))
                
        mean_overall = np.mean(val_f1_scores)
        mean_short = np.mean(short_val_f1_scores)
        
        print(f"  [C={c}] Mean Overall F1: {mean_overall:.4f} | Mean Short F1: {mean_short:.4f}")
        
        if mean_short > best_short_f1:
            best_short_f1 = mean_short
            best_c = c
            
    print(f"\nOptimization Complete! Best C for Short Sentences: {best_c} (Score: {best_short_f1:.4f})")
    
    print(f"Training final model using Best C={best_c} on full dataset...")
    clf_final = LinearSVC(C=best_c, class_weight='balanced', random_state=42, dual=False)
    clf_final.fit(X_train_vec, y_train)
    
    print("Predicting on test set...")
    X_test_vec = vectorizer.transform(X_test)
    test_preds = clf_final.predict(X_test_vec)
    
    print("Generating submission...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sub_path = os.path.join(base_dir, 'output', 'submission_baseline_tuned.csv')
    
    sub = pd.DataFrame({'id': test_df['id'], 'dialect': test_preds})
    sub.to_csv(sub_path, index=False)
    print(f"Tuned baseline submission saved to '{sub_path}'.")

if __name__ == "__main__":
    run_tuned_baseline()
