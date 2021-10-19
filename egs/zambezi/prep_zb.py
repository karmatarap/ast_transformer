# -*- coding: utf-8 -*-
# @Time    : 6/23/21 3:19 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : prep_sc.py
import pandas as pd
import numpy as np
import json
import os

output_path="data"

def load_dz_data(base_data_dir, target_col="age"):
    """Load the Dzanga Bai data from the Excel spreadsheet into a dataframe.

    Also add the spectrogram path, rumble id, and age category columns.

    Additionally, remove any rumbles that are missing age.
    """
    df = pd.read_excel(
        os.path.join(base_data_dir, "Age-sex calls- Dzanga Bai.xlsx"),
        sheet_name="context",
    )
    # Create spectrogram paths in the dataframe.
    df["wav"] = df["unique_ID"].apply(
        lambda x: os.path.join(base_data_dir, "wav", f"{x}.wav")
    )
    df["exists"] = df["wav"].apply(lambda x: os.path.exists(x))
    df["rumble_id"] = df["unique_ID"].apply(lambda x: int(x.split("_")[1]))

    df["agecat"] = df["age"].apply(
        lambda x: "ad/sa"
        if x in ("ad", "sa")
        else "inf/juv"
        if x in ("inf", "juv")
        else "un"
    )
    df = df[df[target_col] != "un"]
    return df

def write_json(df, type="train"):
    data_json = (df[["wav","labels"]].to_json(orient="records"))
    parsed = json.loads(data_json)
    with open(f"./data/datafiles/zambezi_{type}_data.json","w") as f:
        json.dump(parsed, f, indent=1)
    return json 


def main():
    df = load_dz_data(output_path, target_col="agecat")
 
    df["labels"] = df["agecat"]
    with open(os.path.join(output_path, f"train_indices.csv"), "rt") as f:
        train_indices = np.array([int(index) for index in f.readlines()])
    with open(os.path.join(output_path, f"val_indices.csv"), "rt") as f:
        val_indices = np.array([int(index) for index in f.readlines()])
    with open(os.path.join(output_path, f"test_indices.csv"), "rt") as f:
        test_indices = np.array([int(index) for index in f.readlines()])
    df_train = df[df.index.isin(train_indices)].reset_index(drop=True)

    df_val = df[df.index.isin(train_indices)].reset_index(drop=True)
    df_test = df[df.index.isin(test_indices)].reset_index(drop=True)
    print(len(df), len(df_train), len(df_val), len(df_test))
    write_json(df_train, type="train")
    write_json(df_val, type="valid")
    write_json(df_test, type="eval")

if __name__ == "__main__":
    main()


