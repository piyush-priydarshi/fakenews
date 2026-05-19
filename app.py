"""
FAKE NEWS DETECTION SYSTEM - Live Demo App with Voice Output
=============================================================
Features:
1. Smart text preprocessing (matches training pipeline)
2. Confidence threshold handling (uncertain predictions flagged)
3. Short-text detection with user warnings
4. Dual probability display (Fake% and Real%)
5. Voice output using pyttsx3
"""
import pickle
import os
import re
import sys

# pyrefly: ignore [missing-import]
import pyttsx3

# NLTK for preprocessing (must match training!)
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

STOP_WORDS = set(stopwords.words('english'))
stemmer = PorterStemmer()

# ---- Configuration ----
CONFIDENCE_THRESHOLD = 60  # Below this, prediction is "uncertain"
SHORT_TEXT_THRESHOLD = 4   # Warn if fewer than this many words after cleaning
MODEL_PATH = "models/fake_news_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"


def clean_text(text):
    """
    MUST match the exact same cleaning used during training!
    Steps: lowercase -> remove URLs -> remove numbers -> remove special chars
           -> remove extra spaces -> remove stopwords -> stem words
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
    
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        print("Error: Model files not found! Run notebooks/train_model.py first.")
        return None, None
        
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    
    print("Model loaded successfully.")
    return model, vectorizer


def speak_result(text):
    """Uses pyttsx3 to speak the prediction result out loud."""
    try:
        engine = pyttsx3.init()
        rate = engine.getProperty('rate')
        engine.setProperty('rate', rate - 20)
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"  [Voice Warning: {e}]")


def predict_news(statement, model, vectorizer):
    """
    Predicts if a news statement is FAKE or REAL.
    Shows both probabilities and handles edge cases professionally.
    """
    if not statement.strip():
        print("  Please enter a news statement.")
        return
    
    # Step 1: Clean the text (same pipeline as training)
    cleaned = clean_text(statement)
    word_count = len(cleaned.split()) if cleaned else 0
    
    # Step 2: Short-text warning
    if word_count < SHORT_TEXT_THRESHOLD:
        print("\n  [WARNING] Your input is very short after processing.")
        print("  For better accuracy, enter a more detailed news statement.")
        print("  Example: 'Scientists at NASA discover water on Mars surface'\n")
    
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
    
    # Truncate long inputs for display
    display_text = statement[:80] + "..." if len(statement) > 80 else statement
    print(f"  Input: \"{display_text}\"")
    print(f"  Processed Words: {word_count}")
    print(f"  Fake Probability:  {fake_prob:.1f}%")
    print(f"  Real Probability:  {real_prob:.1f}%")
    print(f"  Confidence:        {confidence:.1f}%")
    
    # Step 6: Handle uncertain predictions
    if confidence < CONFIDENCE_THRESHOLD:
        print(f"\n  RESULT: UNCERTAIN")
        print(f"  The AI is not confident enough to make a reliable prediction.")
        print(f"  Try providing more details or a longer news statement.")
        voice_msg = "The prediction is uncertain. Please provide more details for a reliable result."
    else:
        result_label = prediction.upper()
        print(f"\n  RESULT: This news is likely {result_label}")
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
    print(f"  Confidence Threshold: {CONFIDENCE_THRESHOLD}%")
    print(f"  Short Text Warning: <{SHORT_TEXT_THRESHOLD} words")
    
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
