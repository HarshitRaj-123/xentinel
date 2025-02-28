import pandas as pd
import sys
import os

# Add the ai_model folder to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.feature_engineering import preprocess_text

# Load and preprocess each dataset
def load_and_preprocess(file_path):
    data = pd.read_csv(file_path)
    data["processed_text"] = data["Crime Description"].apply(preprocess_text)
    return data

# Combine all datasets
def combine_datasets():
    datasets = [
        "data/raw/crime_dataset_punjab_unique.csv",
    ]
    
    combined_data = pd.concat([load_and_preprocess(file) for file in datasets], ignore_index=True)
    return combined_data

# Save processed data
def save_processed_data(data):
    os.makedirs("data/processed", exist_ok=True)
    data.to_csv("data/processed/combined_crime_data.csv", index=False)

if __name__ == "__main__":
    combined_data = combine_datasets()
    save_processed_data(combined_data)