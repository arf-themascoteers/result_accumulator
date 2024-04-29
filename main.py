import pandas as pd
import os
import re

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def extract_ending_integer(s):
    # Search for one or more digits at the end of the string
    match = re.search(r'\d+$', s)
    if match:
        return int(match.group())
    else:
        return None

final_df = pd.DataFrame(columns=["N","Best_M", "R2_val", "RMSE_val", "Bands"])

for index, file in enumerate(os.listdir("original")):
    counter = file.split("_")[-1].split(".")[0]
    counter = extract_ending_integer(counter)
    path = os.path.join("original", file)
    df = pd.read_csv(path)
    df.drop(["dataset","target_size","fold","algorithm","final_size","time"], axis=1, inplace=True)
    df["repeat"] = df["repeat"]+1
    df.rename(columns={'repeat': 'M', "metric1":"R2_val", "metric2":"RMSE_val", "selected_features":"bands"}, inplace=True)
    df.to_csv(f"modified/{counter}.csv",index=False)


modified_files = os.listdir("modified")
modified_files = sorted(modified_files, key=natural_sort_key)

for index, file in enumerate(modified_files):
    counter = int(file.split(".")[0])
    path = os.path.join("modified", file)
    df = pd.read_csv(path)
    index_of_max_score = df['R2_val'].idxmax()
    row_with_max_score = df.loc[index_of_max_score]
    r2 = row_with_max_score['R2_val']
    rmse = row_with_max_score['RMSE_val']
    N = counter
    M = index_of_max_score + 1
    bands = row_with_max_score['bands']
    final_df.loc[len(final_df)] = {"N":N, "Best_M" : M, "R2_val" : r2, "RMSE_val":rmse, "Bands":bands}

final_df.to_csv(f"out/final.csv",index=False)
