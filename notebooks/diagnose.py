"""
DIAGNOSTIC SCRIPT: Find out WHY predictions are unreliable
This script does NOT change anything - it only investigates.
"""
import pandas as pd
import pickle
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# DIAGNOSIS 1: Check Label Mapping
# ============================================================
print("=" * 60)
print("DIAGNOSIS 1: LABEL MAPPING CHECK")
print("=" * 60)

df = pd.read_csv("../dataset/dataset/fake news detection(FakeNewsNet)/fnn_train.csv")
print(f"Unique labels: {df['label_fnn'].unique()}")
print(f"Label distribution:\n{df['label_fnn'].value_counts()}")

# Spot-check: show a known fake and known real sample
print("\n--- Sample FAKE news statement ---")
fake_sample = df[df['label_fnn'] == 'fake']['statement'].iloc[0]
print(f"Label: fake | Text: {fake_sample[:150]}...")

print("\n--- Sample REAL news statement ---")
real_sample = df[df['label_fnn'] == 'real']['statement'].iloc[0]
print(f"Label: real | Text: {real_sample[:150]}...")

# ============================================================
# DIAGNOSIS 2: Training vs Prediction Text Length Mismatch
# ============================================================
print("\n" + "=" * 60)
print("DIAGNOSIS 2: TEXT LENGTH ANALYSIS (ROOT CAUSE!)")
print("=" * 60)

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# What the model was TRAINED on:
df['train_text'] = df['statement'].apply(clean_text) + " " + df['fullText_based_content'].apply(clean_text)
train_lengths = df['train_text'].str.split().str.len()

print(f"Training text word count stats:")
print(f"  Average: {train_lengths.mean():.0f} words")
print(f"  Minimum: {train_lengths.min()} words")
print(f"  Maximum: {train_lengths.max()} words")
print(f"  Median:  {train_lengths.median():.0f} words")

# What the user types during demo:
test_headline = "3000 Pound Great White shark captured in great lakes"
test_words = len(clean_text(test_headline).split())
print(f"\nUser input word count: {test_words} words")
print(f"MISMATCH RATIO: Model trained on ~{train_lengths.mean():.0f} words, user inputs ~{test_words} words")
print(">>> This is likely the ROOT CAUSE of unreliable predictions! <<<")

# What the model was trained on (just statement column):
stmt_lengths = df['statement'].apply(clean_text).str.split().str.len()
print(f"\nStatement-only word count stats:")
print(f"  Average: {stmt_lengths.mean():.0f} words")
print(f"  Median:  {stmt_lengths.median():.0f} words")
print(">>> Statement-only length is MUCH closer to what users type! <<<")

# ============================================================
# DIAGNOSIS 3: Current Model's Confidence on Known Examples
# ============================================================
print("\n" + "=" * 60)
print("DIAGNOSIS 3: CURRENT MODEL PREDICTIONS ON TEST CASES")
print("=" * 60)

with open("../models/fake_news_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("../models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

test_cases = [
    ("3000 Pound Great White shark captured in great lakes", "OBVIOUSLY FAKE"),
    ("Scientists discover high sugar diet is actually good for children", "OBVIOUSLY FAKE"),
    ("President signs new infrastructure bill into law", "LIKELY REAL"),
    ("NASA confirms alien life found on Mars next week", "OBVIOUSLY FAKE"),
    ("Economy adds 200000 jobs in latest monthly report", "LIKELY REAL"),
]

print(f"\nModel classes: {model.classes_}")

for text, expected in test_cases:
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    probs = model.predict_proba(vec)[0]
    fake_prob = probs[list(model.classes_).index('fake')] * 100
    real_prob = probs[list(model.classes_).index('real')] * 100
    
    status = "CORRECT" if (expected.startswith("OBVIOUSLY FAKE") and pred == "fake") or (expected.startswith("LIKELY REAL") and pred == "real") else "WRONG"
    print(f"\n  Input: \"{text}\"")
    print(f"  Expected: {expected} | Predicted: {pred.upper()} | Fake: {fake_prob:.1f}% | Real: {real_prob:.1f}% | {status}")

# ============================================================
# DIAGNOSIS 4: What if we train on STATEMENT ONLY?
# ============================================================
print("\n" + "=" * 60)
print("DIAGNOSIS 4: STATEMENT-ONLY MODEL COMPARISON")
print("=" * 60)

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load all data
df2 = pd.concat([
    pd.read_csv("../dataset/dataset/fake news detection(FakeNewsNet)/fnn_train.csv"),
    pd.read_csv("../dataset/dataset/fake news detection(FakeNewsNet)/fnn_dev.csv"),
    pd.read_csv("../dataset/dataset/fake news detection(FakeNewsNet)/fnn_test.csv"),
], ignore_index=True)

df2['clean_stmt'] = df2['statement'].apply(clean_text)
df2 = df2[df2['clean_stmt'].str.len() > 5]

X = df2['clean_stmt']
y = df2['label_fnn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vec2 = TfidfVectorizer(stop_words='english', max_df=0.7, min_df=2, ngram_range=(1,2), max_features=30000, sublinear_tf=True)
X_train_v = vec2.fit_transform(X_train)
X_test_v = vec2.transform(X_test)

model2 = LogisticRegression(max_iter=1000, C=10)
model2.fit(X_train_v, y_train)
acc = accuracy_score(y_test, model2.predict(X_test_v))
print(f"Statement-only model accuracy: {acc*100:.2f}%")

print("\nStatement-only model predictions on test cases:")
for text, expected in test_cases:
    cleaned = clean_text(text)
    vec_t = vec2.transform([cleaned])
    pred = model2.predict(vec_t)[0]
    probs = model2.predict_proba(vec_t)[0]
    fake_prob = probs[list(model2.classes_).index('fake')] * 100
    real_prob = probs[list(model2.classes_).index('real')] * 100
    status = "CORRECT" if (expected.startswith("OBVIOUSLY FAKE") and pred == "fake") or (expected.startswith("LIKELY REAL") and pred == "real") else "WRONG"
    print(f"  \"{text[:60]}...\" => {pred.upper()} (Fake:{fake_prob:.1f}% Real:{real_prob:.1f}%) {status}")

print("\n" + "=" * 60)
print("DIAGNOSIS COMPLETE")
print("=" * 60)
