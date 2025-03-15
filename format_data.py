import pandas as pd

def format_quora_data(filepath, output_path):
    df = pd.read_table(filepath)
    df['win'] = df['is_duplicate'].apply(lambda row: 1.0 if row == 1 else 0.0)
    df['lose'] = df['win'].apply(lambda row: 1.0 if row == 0 else 0.0)
    df.to_csv(output_path, index=False)

def merge_data(filepath1, filepath2, output_path):
    df1 = pd.read_csv(filepath1)
    df2 = pd.read_csv(filepath2)
    df = pd.concat([df1, df2], ignore_index=True)
    df.to_csv(output_path, index=False)



if __name__ == "__main__":
    format_quora_data("data/quora-dev.csv", "data/quora-dpo-dev.csv")
    format_quora_data("data/quora-train.csv", "data/quora-dpo-train.csv")
    merge_data("data/quora-dpo-dev.csv", "data/quora-dpo-train.csv", "data/quora-dpo.csv")
