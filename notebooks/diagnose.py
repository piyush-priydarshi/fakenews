"""
LABEL & PREPROCESSING MISMATCH DEBUGGER
========================================
Checks for:
1. Label mapping in dataset
2. Label mapping in saved model
3. Preprocessing mismatch between app.py and training
4. Test predictions on known examples
"""
import pandas as pd
import pickle
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# TASK 1: Inspect dataset labels
# ============================================================
print("=" * 60)
print("TASK 1: DATASET LABEL INSPECTION")
print("=" * 60)

df = pd.read_csv("../dataset/dataset/fake news detection(FakeNewsNet)/fnn_train.csv")
print(f"Unique labels: {df['label_fnn'].unique().tolist()}")
print(f"Label counts:\n{df['label_fnn'].value_counts()}")

# Show specific samples with their labels
print("\n--- 3 samples labeled 'fake' ---")
for i, row in df[df['label_fnn'] == 'fake'].head(3).iterrows():
    print(f"  Label: {row['label_fnn']} | Statement: \"{row['statement'][:100]}\"")

print("\n--- 3 samples labeled 'real' ---")
for i, row in df[df['label_fnn'] == 'real'].head(3).iterrows():
    print(f"  Label: {row['label_fnn']} | Statement: \"{row['statement'][:100]}\"")

# ============================================================
# TASK 2: Check saved model's class mapping
# ============================================================
print("\n" + "=" * 60)
print("TASK 2: SAVED MODEL CLASS MAPPING")
print("=" * 60)

with open("../models/fake_news_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("../models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

print(f"model.classes_ = {model.classes_}")
print(f"  Index 0 = '{model.classes_[0]}'")
print(f"  Index 1 = '{model.classes_[1]}'")
print(f"\nWhen model.predict_proba() returns [p0, p1]:")
print(f"  p0 = probability of '{model.classes_[0]}'")
print(f"  p1 = probability of '{model.classes_[1]}'")

# ============================================================
# TASK 3: Check preprocessing mismatch
# ============================================================
print("\n" + "=" * 60)
print("TASK 3: PREPROCESSING MISMATCH CHECK")
print("=" * 60)

# What the MODEL was trained with (NLTK stemming + stopwords):
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
STOP_WORDS = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text_TRAINING(text):
    """What train_model.py uses"""
    if pd.isna(text) or not text:
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [stemmer.stem(w) for w in text.split() if w not in STOP_WORDS and len(w) > 2]
    return ' '.join(words)

def clean_text_APP(text):
    """What the CURRENT app.py uses (user reverted it!)"""
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

test_input = "ISRO successfully launches satellite into orbit"

training_cleaned = clean_text_TRAINING(test_input)
app_cleaned = clean_text_APP(test_input)

print(f"Original:         \"{test_input}\"")
print(f"Training cleaning: \"{training_cleaned}\"")
print(f"App.py cleaning:   \"{app_cleaned}\"")
print(f"\nMATCH: {'YES' if training_cleaned == app_cleaned else 'NO - MISMATCH FOUND!'}")

if training_cleaned != app_cleaned:
    print("\n  >>> BUG FOUND: app.py uses DIFFERENT text cleaning than train_model.py!")
    print("  >>> The model was trained with NLTK stemming + stopword removal,")
    print("  >>> but app.py does NOT apply these. This causes wrong predictions!")

# ============================================================
# TASK 4: Test predictions with CORRECT vs WRONG preprocessing
# ============================================================
print("\n" + "=" * 60)
print("TASK 4: PREDICTION COMPARISON (correct vs wrong preprocessing)")
print("=" * 60)

test_cases = [
    ("ISRO successfully launches satellite into orbit", "REAL"),
    ("RBI announces new monetary policy keeping repo rate unchanged", "REAL"),
    ("3000 Pound Great White shark captured in great lakes", "FAKE"),
    ("COVID vaccine contains microchips to track citizens", "FAKE"),
    ("Election Commission announces schedule for state elections", "REAL"),
    ("Aliens discovered living under Taj Mahal", "FAKE"),
    ("Chocolate cures cancer says new study", "FAKE"),
]

print(f"\n{'Input':<55} {'Expected':<8} {'Correct Prep':<15} {'Wrong Prep':<15}")
print("-" * 95)

for text, expected in test_cases:
    # With CORRECT preprocessing (matching training)
    correct_clean = clean_text_TRAINING(text)
    vec_correct = vectorizer.transform([correct_clean])
    pred_correct = model.predict(vec_correct)[0]
    probs_correct = model.predict_proba(vec_correct)[0]
    fi = list(model.classes_).index('fake')
    ri = list(model.classes_).index('real')
    
    # With WRONG preprocessing (what current app.py does)
    wrong_clean = clean_text_APP(text)
    vec_wrong = vectorizer.transform([wrong_clean])
    pred_wrong = model.predict(vec_wrong)[0]
    probs_wrong = model.predict_proba(vec_wrong)[0]
    
    c_str = f"{pred_correct.upper()} F:{probs_correct[fi]*100:.0f}%"
    w_str = f"{pred_wrong.upper()} F:{probs_wrong[fi]*100:.0f}%"
    
    print(f"{text[:53]:<55} {expected:<8} {c_str:<15} {w_str:<15}")

print("\n" + "=" * 60)
print("DIAGNOSIS COMPLETE")
print("=" * 60)
