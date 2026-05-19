import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import sys

# Fix encoding issues in windows terminal
sys.stdout.reconfigure(encoding='utf-8')

def main():
    print("Step 1: Loading dataset...")
    data_path = "../dataset/dataset/fake news detection(FakeNewsNet)/fnn_train.csv"
    
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {data_path}. Please check the path.")
        return

    # To improve accuracy, we combine the 'statement' and the 'paragraph_based_content' 
    # to give the AI much more context to learn from.
    df['text'] = df['statement'].fillna('') + " " + df['paragraph_based_content'].fillna('')
    df.dropna(subset=['text', 'label_fnn'], inplace=True)

    X = df['text']
    y = df['label_fnn']

    print(f"Dataset loaded successfully! Total records: {len(df)}")

    # Step 3: Train-Test Split (80% training, 20% testing)
    print("Splitting dataset into 80% training and 20% testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 4: Text Vectorization (TF-IDF)
    # Adding bigrams (ngram_range=(1,2)) to capture phrases like "fake news" or "not true"
    print("Converting text to numerical data using TF-IDF (with N-grams)...")
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, ngram_range=(1,2), max_features=25000)
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Step 5: Train the Random Forest Model
    # We switch to Random Forest for better accuracy, precision, and robust confidence scoring.
    print("Training the Advanced Random Forest AI model...")
    model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    model.fit(X_train_tfidf, y_train)

    # Step 6: Evaluate the Model
    print("Evaluating the model's accuracy...")
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Accuracy: {accuracy * 100:.2f}%\n")
    print("Classification Report (Precision, Recall, F1-Score):")
    print(classification_report(y_test, y_pred))

    # Step 7: Plot Confusion Matrix and Save
    print("Generating confusion matrix chart...")
    cm = confusion_matrix(y_test, y_pred, labels=['fake', 'real'])
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.title('Fake News Detection - Confusion Matrix (Random Forest)')
    plt.tight_layout()
    
    os.makedirs('../outputs', exist_ok=True)
    plt.savefig('../outputs/confusion_matrix.png')
    print("Saved confusion matrix chart to 'outputs/confusion_matrix.png'.")

    # Step 8: Save Model and Vectorizer
    print("Saving the trained model and vectorizer...")
    os.makedirs('../models', exist_ok=True)
    
    with open('../models/fake_news_model.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    with open('../models/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
        
    print("All done! High-Accuracy Model and Vectorizer saved successfully.")

if __name__ == "__main__":
    main()
