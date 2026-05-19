import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import sys
sys.stdout.reconfigure(encoding='utf-8')

data_path = "../dataset/dataset/fake news detection(FakeNewsNet)/fnn_train.csv"
df = pd.read_csv(data_path)

# Let's use the full text content instead of just the statement for much better accuracy!
df['text'] = df['statement'].fillna('') + " " + df['paragraph_based_content'].fillna('')
df.dropna(subset=['text', 'label_fnn'], inplace=True)

X = df['text']
y = df['label_fnn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Adding bigrams (ngram_range=(1,2)) to capture phrases like "fake news" or "not true"
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, ngram_range=(1,2), max_features=25000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("--- Advanced Logistic Regression ---")
model_lr = LogisticRegression(max_iter=1000, C=10)
model_lr.fit(X_train_tfidf, y_train)
y_pred_lr = model_lr.predict(X_test_tfidf)
print("LR Accuracy:", accuracy_score(y_test, y_pred_lr))
print("LR F1 Score:")
print(classification_report(y_test, y_pred_lr))

print("--- Random Forest ---")
model_rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
model_rf.fit(X_train_tfidf, y_train)
y_pred_rf = model_rf.predict(X_test_tfidf)
print("RF Accuracy:", accuracy_score(y_test, y_pred_rf))
print("RF F1 Score:")
print(classification_report(y_test, y_pred_rf))
