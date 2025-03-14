import pandas as pd

def format_quora_data(filepath, output_path):
    df = pd.read_table(filepath)
    df['win'] = df['is_duplicate'].apply(lambda row: 1.0 if row == 1 else 0.0)
    df['lose'] = df['win'].apply(lambda row: 1.0 if row == 0 else 0.0)
    df.to_csv(output_path, index=False)



if __name__ == "__main__":
    format_quora_data("data/quora-dev.csv", "data/quora-dpo-dev.csv")