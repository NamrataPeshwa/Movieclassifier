import pandas as pd
from classifier import preprocess_text, extract_features, train_model #importing the classifier file 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def load_train_data(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(" ::: ")
            if len(parts) == 4:
                data.append({
                    'id': parts[0],
                    'title': parts[1],
                    'genre': parts[2],
                    'description': parts[3]
                })
    return pd.DataFrame(data)

def evaluate(path):
    print("Loading and preprocessing training data...")
    df = load_train_data(path)
    df['clean_description'] = df['description'].apply(preprocess_text)

    print("Splitting into training and validation sets...")
    X = df['clean_description']
    y = df['genre']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_tfidf, X_val_tfidf, _ = extract_features(X_train, X_val)
    model = train_model(X_train_tfidf, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_val_tfidf)

    print("Classification Report:\n")
    print(classification_report(y_val, y_pred))

if __name__ == "__main__":
    evaluate(r"E:/Namrata/programming/using git/Movieclassifier/movieclassifier/train_data.txt")
