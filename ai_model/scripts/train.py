# scripts/train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
import joblib

# Load processed data
def load_data():
    data = pd.read_csv("data/processed/combined_crime_data.csv")
    return data

# Train the model
def train_model():
    # Load and preprocess data
    data = load_data()
    X = data[["processed_text", "Location", "Time", "Crime Severity"]]  # Use multiple features
    y = data["Crime Type"]  # Target variable
    
    # Preprocess text and categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(max_features=5000), "processed_text"),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["Location", "Time", "Crime Severity"])
        ]
    )
    
    X_transformed = preprocessor.fit_transform(X)
    
    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_transformed, y)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
    
    # Train a Logistic Regression classifier
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    # Save the model and preprocessor
    joblib.dump(model, "models/crime_model.pkl")
    joblib.dump(preprocessor, "models/preprocessor.pkl")

if __name__ == "__main__":
    train_model()