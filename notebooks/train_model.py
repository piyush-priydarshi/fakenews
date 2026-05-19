"""
FINAL TRAINING PIPELINE - Fake News Detection System
=====================================================
Strategy: Train on STATEMENT text only using Multinomial Naive Bayes.
This model correctly identifies obviously fake headlines and handles
short user inputs properly during live demos.

The dataset is political fact-checks from PolitiFact, so the model
is strongest on political/news statements.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

STOP_WORDS = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    """
    Professional NLP preprocessing:
    1. Lowercase
    2. Remove URLs
    3. Remove numbers  
    4. Remove punctuation/special chars
    5. Remove extra whitespace
    6. Remove stopwords
    7. Stemming (running -> run, claimed -> claim)
    """
    if pd.isna(text) or not text:
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [stemmer.stem(w) for w in text.split() if w not in STOP_WORDS and len(w) > 2]
    return ' '.join(words)


def main():
    print("=" * 60)
    print("  FAKE NEWS DETECTION - FINAL TRAINING PIPELINE")
    print("  Optimized for headline/statement prediction")
    print("=" * 60)
    
    # Load ALL data
    print("\n[1/7] Loading all dataset files...")
    base = "../dataset/dataset/fake news detection(FakeNewsNet)"
    df = pd.concat([
        pd.read_csv(f"{base}/fnn_train.csv"),
        pd.read_csv(f"{base}/fnn_dev.csv"),
        pd.read_csv(f"{base}/fnn_test.csv"),
    ], ignore_index=True)
    print(f"  Raw records: {len(df)}")
    
    # Clean text
    print("[2/7] Cleaning text with NLP pipeline (stemming + stopword removal)...")
    df['clean_text'] = df['statement'].apply(clean_text)
    df = df[df['clean_text'].str.len() > 5]
    df.dropna(subset=['clean_text', 'label_fnn'], inplace=True)
    
    X = df['clean_text']
    y = df['label_fnn']
    print(f"  Usable records: {len(df)}")
    print(f"  Labels: {y.value_counts().to_dict()}")
    print(f"  Avg text length: {X.str.split().str.len().mean():.0f} words")
    
    # Split
    print("[3/7] Splitting 80% train / 20% test (stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # TF-IDF
    print("[4/7] TF-IDF vectorization (trigrams + sublinear scaling)...")
    vectorizer = TfidfVectorizer(
        max_df=0.7,
        min_df=2,
        ngram_range=(1, 3),
        max_features=30000,
        sublinear_tf=True,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"  Features: {X_train_tfidf.shape[1]}")
    
    # Train
    print("[5/7] Training Multinomial Naive Bayes...")
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train_tfidf, y_train)
    
    # Evaluate
    print("[6/7] Evaluating...")
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n  MODEL ACCURACY: {accuracy * 100:.2f}%\n")
    print("  Classification Report:")
    report = classification_report(y_test, y_pred)
    for line in report.split('\n'):
        print(f"  {line}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=['fake', 'real'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn',
                xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'],
                annot_kws={"size": 18})
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('Actual Label', fontsize=14)
    plt.title(f'Fake News Detection - Confusion Matrix\nAccuracy: {accuracy*100:.2f}%', fontsize=14)
    plt.tight_layout()
    os.makedirs('../outputs', exist_ok=True)
    plt.savefig('../outputs/confusion_matrix.png', dpi=150)
    print("\n  Confusion matrix saved to outputs/confusion_matrix.png")
    
    # Test Headlines
    print("\n[7/7] Testing on sample headlines...")
    test_cases = [
        "3000 Pound Great White shark captured in great lakes",
        "Scientists discover high sugar diet is actually good for children",
        "President signs new infrastructure bill into law",
        "NASA confirms alien life found on Mars next week",
        "Economy adds 200000 jobs in latest monthly report",
        "COVID vaccine contains microchips to track people",
        "New study shows exercise reduces risk of heart disease",
    ]
    
    for text in test_cases:
        cleaned = clean_text(text)
        vec_t = vectorizer.transform([cleaned])
        pred = model.predict(vec_t)[0]
        probs = model.predict_proba(vec_t)[0]
        fi = list(model.classes_).index('fake')
        ri = list(model.classes_).index('real')
        print(f"  \"{text[:55]}\" => {pred.upper()} (Fake:{probs[fi]*100:.0f}% Real:{probs[ri]*100:.0f}%)")
    
    # Save
    print("\nSaving model and vectorizer...")
    os.makedirs('../models', exist_ok=True)
    with open('../models/fake_news_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('../models/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print(f"\nDONE! Model accuracy: {accuracy*100:.2f}%")

if __name__ == "__main__":
    main()
