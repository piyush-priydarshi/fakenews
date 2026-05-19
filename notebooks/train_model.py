import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

def clean_text(text):
    """Aggressive text cleaning for better NLP results."""
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove all non-letter characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text

def main():
    print("Step 1: Loading ALL dataset files for maximum training data...")
    base = "../dataset/dataset/fake news detection(FakeNewsNet)"

    # Load ALL three CSV files to maximize training data
    df_train = pd.read_csv(f"{base}/fnn_train.csv")
    df_dev = pd.read_csv(f"{base}/fnn_dev.csv")
    df_test = pd.read_csv(f"{base}/fnn_test.csv")

    # Combine them all into one big dataset
    df_all = pd.concat([df_train, df_dev, df_test], ignore_index=True)
    print(f"Combined dataset size: {len(df_all)} records")

    # Use FULL article text + statement for maximum context
    # This is the single biggest accuracy booster!
    df_all['clean_text'] = df_all['statement'].apply(clean_text) + " " + df_all['fullText_based_content'].apply(clean_text)
    df_all.dropna(subset=['clean_text', 'label_fnn'], inplace=True)
    df_all = df_all[df_all['clean_text'].str.len() > 10]

    X = df_all['clean_text']
    y = df_all['label_fnn']

    print(f"Total usable records after cleaning: {len(df_all)}")
    print(f"Label distribution:\n{y.value_counts()}")

    # Step 2: Train-Test Split
    print("\nSplitting into 80% training and 20% testing...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 3: Advanced TF-IDF Vectorization
    # - sublinear_tf=True: applies log scaling (big accuracy boost for text)
    # - ngram_range=(1,2): captures word pairs like "fake news"
    # - max_features=50000: uses top 50K most informative words
    # - min_df=2: ignores words that appear only once (noise)
    print("Converting text with advanced TF-IDF (sublinear + bigrams)...")
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_df=0.7,
        min_df=2,
        ngram_range=(1, 2),
        max_features=50000,
        sublinear_tf=True
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"TF-IDF features created: {X_train_tfidf.shape[1]}")

    # Step 4: Train High-Accuracy Logistic Regression
    # C=10 makes the model fit more tightly to the training data
    print("Training the high-accuracy Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, C=10)
    model.fit(X_train_tfidf, y_train)

    # Step 5: Evaluate
    print("\nEvaluating model performance...")
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"  MODEL ACCURACY: {accuracy * 100:.2f}%")
    print(f"{'='*50}\n")
    print("Detailed Classification Report (Precision, Recall, F1-Score):")
    print(classification_report(y_test, y_pred))

    # Step 6: Confusion Matrix
    print("Generating confusion matrix...")
    cm = confusion_matrix(y_test, y_pred, labels=['fake', 'real'])

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn',
                xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'],
                annot_kws={"size": 16})
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('Actual Label', fontsize=14)
    plt.title(f'Fake News Detection - Confusion Matrix\nAccuracy: {accuracy*100:.2f}%', fontsize=16)
    plt.tight_layout()

    os.makedirs('../outputs', exist_ok=True)
    plt.savefig('../outputs/confusion_matrix.png', dpi=150)
    print("Saved confusion matrix to 'outputs/confusion_matrix.png'.")

    # Step 7: Save Model and Vectorizer
    print("Saving the trained model and vectorizer...")
    os.makedirs('../models', exist_ok=True)

    with open('../models/fake_news_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('../models/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    print("SUCCESS! High-accuracy model saved to models/ folder.")

if __name__ == "__main__":
    main()
