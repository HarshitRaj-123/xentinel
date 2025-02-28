import joblib

# Load the trained model and vectorizer
def load_model():
    model = joblib.load("models/crime_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    return model, vectorizer