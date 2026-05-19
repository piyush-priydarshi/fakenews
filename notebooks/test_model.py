import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

print("Loading dataset...")
data_path = "../dataset/dataset/fake news detection(FakeNewsNet)/fnn_train.csv"
df = pd.read_csv(data_path)

# ===== KEY INSIGHT: Use the FULL article text for maximum accuracy =====
# The 'fullText_based_content' column has the entire article, giving the AI
# much more signal to detect fake vs real news patterns.

def clean_text(text):
    """Aggressive text cleaning for better NLP results."""
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove all non-letters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text

# Combine statement + full text for maximum context
df['clean_text'] = df['statement'].apply(clean_text) + " " + df['fullText_based_content'].apply(clean_text)
df.dropna(subset=['clean_text', 'label_fnn'], inplace=True)
df = df[df['clean_text'].str.len() > 10]  # Remove very short/empty entries

X = df['clean_text']
y = df['label_fnn']

print(f"Total records after cleaning: {len(df)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Aggressive TF-IDF with sublinear scaling and bigrams
print("Vectorizing with advanced TF-IDF...")
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.7,
    min_df=2,
    ngram_range=(1, 2),
    max_features=50000,
    sublinear_tf=True  # Applies log scaling - big accuracy boost for text!
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"TF-IDF features: {X_train_tfidf.shape[1]}")
print("=" * 60)

# Test multiple models to find the best one
models = {
    "Logistic Regression (C=10)": LogisticRegression(max_iter=1000, C=10),
    "Passive Aggressive Classifier": PassiveAggressiveClassifier(max_iter=100, C=0.5),
    "Linear SVC": LinearSVC(max_iter=2000, C=1.0),
    "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
}

best_acc = 0
best_name = ""

for name, model in models.items():
    print(f"\n--- {name} ---")
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc * 100:.2f}%")
    print(classification_report(y_test, y_pred))
    if acc > best_acc:
        best_acc = acc
        best_name = name

print("=" * 60)
print(f"BEST MODEL: {best_name} with {best_acc * 100:.2f}% accuracy")
