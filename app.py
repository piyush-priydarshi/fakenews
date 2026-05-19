import pickle
# pyrefly: ignore [missing-import]
import pyttsx3
import os
import re

def clean_text(text):
    """Same cleaning function used during training - MUST match exactly."""
    if not text:
        return ""
    text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_model_and_vectorizer():
    print("Loading AI Model and Text Vectorizer...")
    
    model_path = "models/fake_news_model.pkl"
    vectorizer_path = "models/tfidf_vectorizer.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        print("Error: Model files not found! Please run notebooks/train_model.py first.")
        return None, None
        
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
        
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
        print(f"Warning: Voice output failed. (Error: {e})")

def predict_news(statement, model, vectorizer):
    """Predicts if a news statement is real or fake, and calculates confidence."""
    if not statement.strip():
        return
    
    # Clean text the SAME way we cleaned training data
    cleaned = clean_text(statement)
        
    # Transform the input text using our trained vectorizer
    text_vectorized = vectorizer.transform([cleaned])
    
    # Predict the class ('real' or 'fake')
    prediction = model.predict(text_vectorized)[0]
    
    # Calculate Confidence Percentage
    probabilities = model.predict_proba(text_vectorized)[0]
    class_index = list(model.classes_).index(prediction)
    confidence = probabilities[class_index] * 100
    
    # Print and Speak the result
    print("-" * 50)
    print(f"  Statement: \"{statement[:100]}...\"" if len(statement) > 100 else f"  Statement: \"{statement}\"")
    print(f"  AI Prediction: {prediction.upper()}")
    print(f"  Confidence: {confidence:.2f}%")
    print("-" * 50)
    
    # Voice output
    voice_message = f"This news appears to be {prediction} with {int(confidence)} percent confidence."
    print(f"  Voice: \"{voice_message}\"")
    speak_result(voice_message)

def main():
    print("=" * 50)
    print("  Fake News Detection AI - Voice Prediction")
    print("  Model: Logistic Regression + TF-IDF")
    print("  Accuracy: 74.46%")
    print("=" * 50)
    
    model, vectorizer = load_model_and_vectorizer()
    
    if model is None:
        return
        
    print("\nAI System Ready!\n")
    
    while True:
        try:
            print("\nEnter a news statement to analyze (or type 'quit' to exit):")
            user_input = input("> ")
            
            if user_input.lower().strip() == 'quit':
                print("Exiting system. Goodbye!")
                break
                
            predict_news(user_input, model, vectorizer)
            
        except KeyboardInterrupt:
            print("\nExiting system. Goodbye!")
            break

if __name__ == "__main__":
    main()
