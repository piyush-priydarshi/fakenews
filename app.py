"""
FAKE NEWS DETECTION SYSTEM - Live Demo App with Voice Output
=============================================================
IMPORTANT: Text cleaning here MUST exactly match train_model.py
"""
import pickle
# pyrefly: ignore [missing-import]
import pyttsx3
import os
import re

# NLTK preprocessing (MUST match training pipeline exactly!)
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

STOP_WORDS = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Configuration
CONFIDENCE_THRESHOLD = 60  # Below this = UNCERTAIN
SHORT_TEXT_THRESHOLD = 3   # Warn if fewer words after cleaning


def clean_text(text):
    """
    Professional NLP cleaning - MUST match train_model.py exactly!
    1. Lowercase
    2. Remove URLs
    3. Remove numbers
    4. Remove special characters
    5. Remove extra whitespace
    6. Remove stopwords
    7. Stemming (running -> run, launches -> launch)
    """
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [stemmer.stem(w) for w in text.split() if w not in STOP_WORDS and len(w) > 2]
    return ' '.join(words)


def load_model_and_vectorizer():
    """Loads the trained model and TF-IDF vectorizer from disk."""
    print("Loading AI Model and Text Vectorizer...")
    model_path = "models/fake_news_model.pkl"
    vectorizer_path = "models/tfidf_vectorizer.pkl"

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        print("Error: Model files not found! Run notebooks/train_model.py first.")
        return None, None

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)

    print(f"Model loaded. Classes: {model.classes_}")
    return model, vectorizer


def speak_result(text):
    """Uses pyttsx3 to speak the text out loud."""
    try:
        engine = pyttsx3.init()
        rate = engine.getProperty('rate')
        engine.setProperty('rate', rate - 20)
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"  [Voice warning: {e}]")


def predict_news(statement, model, vectorizer):
    """Predicts if news is FAKE or REAL with confidence handling."""
    if not statement.strip():
        print("  Please enter a news statement.")
        return

    # Step 1: Clean text (MUST match training!)
    cleaned = clean_text(statement)
    word_count = len(cleaned.split()) if cleaned else 0

    # Step 2: Short-text warning
    if word_count < SHORT_TEXT_THRESHOLD:
        print("\n  [WARNING] Input is very short after processing.")
        print("  For better accuracy, provide a more detailed statement.")

    # Step 3: Vectorize and predict
    text_vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(text_vectorized)[0]

    # Step 4: Get probabilities for BOTH classes
    probabilities = model.predict_proba(text_vectorized)[0]
    fake_index = list(model.classes_).index('fake')
    real_index = list(model.classes_).index('real')
    fake_prob = probabilities[fake_index] * 100
    real_prob = probabilities[real_index] * 100
    confidence = max(fake_prob, real_prob)

    # Step 5: Display results
    print("\n" + "=" * 50)
    display_text = statement[:80] + "..." if len(statement) > 80 else statement
    print(f"  Input:   \"{display_text}\"")
    print(f"  Cleaned: \"{cleaned[:60]}\"")
    print(f"  Fake Probability:  {fake_prob:.1f}%")
    print(f"  Real Probability:  {real_prob:.1f}%")
    print(f"  Confidence:        {confidence:.1f}%")

    # Step 6: Confidence-based result
    if confidence < CONFIDENCE_THRESHOLD:
        print(f"\n  RESULT: UNCERTAIN")
        print(f"  AI is not confident enough. Try a more detailed statement.")
        voice_msg = "The prediction is uncertain. Please provide more details."
    else:
        print(f"\n  RESULT: This news is likely {prediction.upper()}")
        voice_msg = f"This news appears to be {prediction} with {int(confidence)} percent confidence."

    print("=" * 50)

    # Step 7: Voice output
    print(f"  Speaking: \"{voice_msg}\"")
    speak_result(voice_msg)


def main():
    print("=" * 50)
    print("  FAKE NEWS DETECTION AI")
    print("  Voice-Powered Prediction System")
    print("=" * 50)

    model, vectorizer = load_model_and_vectorizer()
    if model is None:
        return

    print("\nSystem Ready! Type a news headline to analyze.\n")

    while True:
        try:
            print("Enter news statement (or 'quit' to exit):")
            user_input = input("> ")

            if user_input.lower().strip() in ('quit', 'exit', 'q'):
                speak_result("Thank you for using the fake news detection system. Goodbye!")
                print("Goodbye!")
                break

            predict_news(user_input, model, vectorizer)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except EOFError:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
