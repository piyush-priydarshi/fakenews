"""Quick test: statement + paragraph_based_content (shorter than full text)"""
import pandas as pd
import re, sys, nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
sys.stdout.reconfigure(encoding='utf-8')
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

STOP_WORDS = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    if pd.isna(text) or not text:
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [stemmer.stem(w) for w in text.split() if w not in STOP_WORDS and len(w) > 2]
    return ' '.join(words)

base = "../dataset/dataset/fake news detection(FakeNewsNet)"
df = pd.concat([
    pd.read_csv(f"{base}/fnn_train.csv"),
    pd.read_csv(f"{base}/fnn_dev.csv"),
    pd.read_csv(f"{base}/fnn_test.csv"),
], ignore_index=True)

# Strategy: statement + paragraph_based_content (summary, not full article)
df['text'] = df['statement'].apply(clean_text) + " " + df['paragraph_based_content'].apply(clean_text)
df = df[df['text'].str.len() > 5]
X = df['text']
y = df['label_fnn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

vec = TfidfVectorizer(max_df=0.7, min_df=2, ngram_range=(1,3), max_features=50000, sublinear_tf=True)
Xtr = vec.fit_transform(X_train)
Xte = vec.transform(X_test)

models = {
    "LR C=10": LogisticRegression(max_iter=1000, C=10),
    "NB 0.1": MultinomialNB(alpha=0.1),
    "SVC Cal": CalibratedClassifierCV(LinearSVC(max_iter=2000, C=1.0), cv=3),
}

test_cases = [
    ("3000 Pound Great White shark captured in great lakes", "FAKE"),
    ("Scientists discover high sugar diet is actually good for children", "FAKE"),
    ("President signs new infrastructure bill into law", "REAL"),
    ("NASA confirms alien life found on Mars next week", "FAKE"),
    ("Economy adds 200000 jobs in latest monthly report", "REAL"),
]

for name, model in models.items():
    model.fit(Xtr, y_train)
    acc = accuracy_score(y_test, model.predict(Xte))
    print(f"\n{name}: Accuracy {acc*100:.2f}%")
    for text, exp in test_cases:
        cleaned = clean_text(text)
        v = vec.transform([cleaned])
        pred = model.predict(v)[0]
        probs = model.predict_proba(v)[0]
        fi = list(model.classes_).index('fake')
        ri = list(model.classes_).index('real')
        s = "OK" if pred.upper()==exp else "MISS"
        print(f"  [{s}] \"{text[:50]}\" => {pred.upper()} F:{probs[fi]*100:.0f}% R:{probs[ri]*100:.0f}%")
