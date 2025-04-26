import pandas as pd
import re
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download stopwords once
nltk.download('stopwords')
from nltk.corpus import stopwords

# Step 1: Load Data


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

def load_test_data(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(" ::: ")
            if len(parts) == 3:
                data.append({
                    'id': parts[0],
                    'title': parts[1],
                    'description': parts[2]
                })
    return pd.DataFrame(data)

# Step 2: Preprocess Text


stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)


# Step 3: TF-IDF Feature Extraction


def extract_features(train_corpus, test_corpus):
    #Convert cleaned text into TF-IDF vector
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_corpus)
    X_test = vectorizer.transform(test_corpus)
    return X_train, X_test, vectorizer

# Step 4: Train Model

def train_model(X, y):
    """Train a Logistic Regression classifier."""
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model


# Step 5: Predict Genres
# -------------------------------

def predict_genres(model, X_test):
    """Predict genres for the test set."""
    return model.predict(X_test)

# -------------------------------
# Step 6: Save Predictions
# -------------------------------

def save_predictions(test_df, predictions, output_file="predictions.txt"):
    """Write predictions to output file in required format."""
    test_df['predicted_genre'] = predictions
    with open(output_file, 'w', encoding='utf-8') as f:
        for _, row in test_df.iterrows():
            f.write(f"{row['id']} ::: {row['title']} ::: {row['predicted_genre']}\n")
    print(f"\n✅ Predictions saved to '{output_file}'")

# -------------------------------
# Main Workflow
# -------------------------------

def main():
    print("loading data...")
    train_df = load_train_data(r"E:\Namrata\programming\using git\Movieclassifier\movieclassifier\train_data.txt")
    test_df = load_test_data(r"E:\Namrata\programming\using git\Movieclassifier\movieclassifier\test_data.txt")

    print("Preprocessing or cleaning descriptions...")
    train_df['clean_description'] = train_df['description'].apply(preprocess_text)
    test_df['clean_description'] = test_df['description'].apply(preprocess_text)

    print("Extracting features with TF-IDF...")
    X_train, X_test, vectorizer = extract_features(train_df['clean_description'], test_df['clean_description'])

    print("Training Logistic Regression model...")
    model = train_model(X_train, train_df['genre'])

    print("Predicting genres...")
    predictions = predict_genres(model, X_test)

    print("Saving predictions...")
    save_predictions(test_df, predictions)

if __name__ == "__main__":
    main()
