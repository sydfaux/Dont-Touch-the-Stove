import os
import pandas as pd
import re
from sklearn.model_selection import train_test_split



def load_sonnets(file_path):
    """Load and split sonnets from the text file."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Split by sonnet numbering (e.g., "I.", "II.", etc.)
    sonnets = re.split(r"\n\s*(?:\d+|[IVXLCDM]+)\.\s*\n", text)

    # Remove empty entries and clean spaces
    sonnets = [s.strip() for s in sonnets if s.strip()]

    return sonnets

def save_to_csv(df, output_file, append=False):
    """Save DataFrame to CSV (append if needed)."""
    mode = "a" if append else "w"
    header = not os.path.exists(output_file) if append else True  # Only write header if file doesn't exist
    df.to_csv(output_file, index=False, encoding="utf-8", mode=mode, header=header)
    print(f"Appended {len(df)} records to {output_file}" if append else f"Saved {len(df)} records to {output_file}")

def clean_text(text):
    """Remove unwanted characters and normalize spacing."""
    text = re.sub(r"[^\w\s,.'!?;:()]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # Normalize spaces
    return text

def preprocess_data(sonnets):
    """Convert list of sonnets into a cleaned DataFrame."""
    df = pd.DataFrame({"text": sonnets})
    df["text"] = df["text"].apply(clean_text)
    return df

def split_and_save(df, processed_dir):
    """Assign first sonnet to train, second to validation, and append to existing files."""
    if len(df) < 2:
        raise ValueError("Not enough sonnets to split into train and validation.")

    # Assign the first to train and the second to validation
    train_df = df.iloc[:1]
    val_df = df.iloc[1:2]

    # Define file paths
    train_file = os.path.join(processed_dir, "train.csv")
    val_file = os.path.join(processed_dir, "val.csv")
    test_file = os.path.join(processed_dir, "test.csv")

    # Append data to existing files
    save_to_csv(train_df, train_file, append=True)
    save_to_csv(val_df, val_file, append=True)

    # Append remaining sonnets to test set
    if len(df) > 2:
        test_df = df.iloc[2:]
        save_to_csv(test_df, test_file, append=True)

    print(f"Train, validation, and test datasets updated in: {processed_dir}")

if __name__ == "__main__":
        
    # Define input and output file paths
    DATA_DIR = os.path.join(os.getcwd(), "data")
    RAW_DATA_FILE = os.path.join(DATA_DIR, "sonnets", "sonnets.txt") 
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

    # Ensure the processed data directory exists
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    sonnets_list = load_sonnets(RAW_DATA_FILE)  # Use the correct file path

    # Preprocess and split dataset
    df = preprocess_data(sonnets_list)
    # Save structured data to CSV
    output_csv_file = os.path.join(PROCESSED_DATA_DIR, "sonnets.csv")
    save_to_csv(df, output_csv_file)  

    split_and_save(df, PROCESSED_DATA_DIR)