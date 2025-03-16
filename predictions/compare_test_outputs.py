import pandas as pd

def print_diff_outputs(filename1, filename2):
    df1 = pd.read_table(filename1, delimiter=' ').sort_values(by='\t')
    df2 = pd.read_table(filename2, delimiter=' ').sort_values(by='\t')
    k = 0
    for (index1, row1), (index2, row2) in zip(df1.iterrows(), df2.iterrows()):
        if (k == 5):
            return
        if (row1['Predicted_Is_Paraphrase'] != row2['Predicted_Is_Paraphrase']):
            print(row1['id'])
            k+=1

if __name__ == "__main__":
    print_diff_outputs("para-test-output-base.csv", "para-test-output-dpo.csv")