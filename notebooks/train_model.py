import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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
    # Loading the FakeNewsNet dataset we identified earlier
    data_path = "../dataset/dataset/fake news detection(FakeNewsNet)/fnn_train.csv"
    
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {data_path}. Please check the path.")
        return

    # Ensure there are no null values in our input or target columns
    df.dropna(subset=['statement', 'label_fnn'], inplace=True)

    # Step 2: Extract Text and Labels
    # 'statement' is the news text, 'label_fnn' is 'fake' or 'real'
    X = df['statement']
    y = df['label_fnn']

    print(f"Dataset loaded successfully! Total records: {len(df)}")

    # Step 3: Train-Test Split (80% training, 20% testing)
    print("Splitting dataset into 80% training and 20% testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 4: Text Vectorization (TF-IDF)
    # AI models can't read text, so we convert text to numbers using TF-IDF.
    # stop_words='english' removes common words like 'the', 'is', 'in'
    print("Converting text to numerical data using TF-IDF...")
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Step 5: Train the Logistic Regression Model
    # We use Logistic Regression because it's fast, simple, and very effective for text classification.
    print("Training the Logistic Regression AI model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    # Step 6: Evaluate the Model
    print("Evaluating the model's accuracy...")
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Accuracy: {accuracy * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Step 7: Plot Confusion Matrix and Save
    # A Confusion Matrix visually shows us True Positives, False Positives, etc.
    print("Generating confusion matrix chart...")
    cm = confusion_matrix(y_test, y_pred, labels=['fake', 'real'])
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.title('Fake News Detection - Confusion Matrix')
    plt.tight_layout()
    
    os.makedirs('../outputs', exist_ok=True)
    plt.savefig('../outputs/confusion_matrix.png')
    print("Saved confusion matrix chart to 'outputs/confusion_matrix.png'.")

    # Step 8: Save Model and Vectorizer
    # We save these so we can use them in our live demo script later without retraining!
    print("Saving the trained model and vectorizer...")
    os.makedirs('../models', exist_ok=True)
    
    with open('../models/fake_news_model.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    with open('../models/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
        
    print("All done! Model and Vectorizer saved successfully.")

if __name__ == "__main__":
    main()
