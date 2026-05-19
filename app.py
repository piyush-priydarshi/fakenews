import pickle
# pyrefly: ignore [missing-import]
import pyttsx3
import os

def load_model_and_vectorizer():
    print("Loading AI Model and Text Vectorizer...")
    
    # Path to models
    model_path = "models/fake_news_model.pkl"
    vectorizer_path = "models/tfidf_vectorizer.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        print("❌ Error: Model files not found! Please run notebooks/train_model.py first.")
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
        # Set speech rate a bit slower for clarity
        rate = engine.getProperty('rate')
        engine.setProperty('rate', rate - 20)
        
        # Say the text
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Warning: Voice output failed. (Error: {e})")

def predict_news(statement, model, vectorizer):
    """Predicts if a news statement is real or fake, and calculates confidence."""
    if not statement.strip():
        return
        
    # 1. Transform the input text into numerical data using our trained vectorizer
    text_vectorized = vectorizer.transform([statement])
    
    # 2. Predict the class ('real' or 'fake')
    prediction = model.predict(text_vectorized)[0]
    
    # 3. Calculate Confidence Percentage
    # predict_proba returns probabilities for each class
    # Classes are usually sorted alphabetically: ['fake', 'real']
    probabilities = model.predict_proba(text_vectorized)[0]
    class_index = list(model.classes_).index(prediction)
    confidence = probabilities[class_index] * 100
    
    # 4. Print and Speak the result
    print("-" * 50)
    print(f"📰 Statement: \"{statement}\"")
    print(f"🤖 AI Prediction: {prediction.upper()}")
    print(f"📊 Confidence: {confidence:.2f}%")
    print("-" * 50)
    
    # Voice output string
    voice_message = f"This news appears to be {prediction} with {int(confidence)} percent confidence."
    speak_result(voice_message)

def main():
    print("=" * 50)
    print("🎙️ Welcome to the Fake News Detection AI 🎙️")
    print("=" * 50)
    
    model, vectorizer = load_model_and_vectorizer()
    
    if model is None:
        return
        
    print("\n✅ AI System Ready!\n")
    
    # Interactive loop for the live demo
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
