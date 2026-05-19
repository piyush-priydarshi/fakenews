import pandas as pd
import sys

def analyze(file_path):
    print(f"--- Analyzing {file_path} ---")
    try:
        df = pd.read_csv(file_path)
        print("Columns:", df.columns.tolist())
        print("Shape:", df.shape)
        print("Missing values:\n", df.isnull().sum())
        
        # Try to guess label col
        label_cols = [c for c in df.columns if 'label' in c.lower() or 'class' in c.lower() or 'target' in c.lower() or c == 'fake' or c == 'real']
        if label_cols:
            print(f"Distribution of {label_cols[0]}:\n", df[label_cols[0]].value_counts())
        else:
            print("No obvious label column found. Printing first 2 rows:\n", df.head(2))
    except Exception as e:
        print("Error:", e)
    print("\n")

analyze("dataset/dataset/fake news detection(FakeNewsNet)/fnn_train.csv")
analyze("dataset/dataset/fake news detection(LIAR)/liar_train.csv")
