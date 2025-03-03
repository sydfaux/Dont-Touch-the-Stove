import kagglehub
import os
import pandas as pd
import re
from sklearn.model_selection import train_test_split

def clean_text(text):
    """Remove unwanted characters and normalize spacing."""
    text = re.sub(r"[^\w\s,.'!?;:()]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # Normalize spaces
    return text

def load_sonnet_texts(directory):
    """Load all sonnet `.txt` files from the directory."""
    sonnets = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read().strip()
                sonnets.append(clean_text(text))
    return sonnets

def preprocess_data():
    """Load, clean, and split the dataset from text files."""
    # Define dataset directory
    raw_sonnet_dir = os.path.join(os.getcwd(), "data", "poems", "forms", "sonnet")
    
    if not os.path.exists(raw_sonnet_dir):
        print(f"Dataset directory not found: {raw_sonnet_dir}")
        return
    
    # Load all sonnets
    sonnets = load_sonnet_texts(raw_sonnet_dir)
    
    if not sonnets:
        print("No sonnet text files found.")
        return

    # Create DataFrame
    df = pd.DataFrame({"text": sonnets})

    # Create new directory for processed data
    PROCESSED_DATA_DIR = os.path.join(os.getcwd(), "data", "processed")
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # Split into train, validation, and test sets
    train_texts, temp_texts = train_test_split(df["text"], test_size=0.2, random_state=42)
    val_texts, test_texts = train_test_split(temp_texts, test_size=0.5, random_state=42)

    # Save processed datasets
    train_texts.to_csv(os.path.join(PROCESSED_DATA_DIR, "train.csv"), index=False)
    val_texts.to_csv(os.path.join(PROCESSED_DATA_DIR, "val.csv"), index=False)
    test_texts.to_csv(os.path.join(PROCESSED_DATA_DIR, "test.csv"), index=False)

    print("Preprocessing complete. Processed files saved in:", PROCESSED_DATA_DIR)

if __name__ == "__main__":
    preprocess_data()