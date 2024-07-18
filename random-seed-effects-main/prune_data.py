import argparse
from collections import Counter
from pathlib import Path
import pandas as pd
from static import *


def prune_data(data_set_name, prune_technique):
    # load the data
    data = pd.read_csv(f"./{DATA_FOLDER}/{data_set_name}/{CLEAN_FOLDER}/{CLEAN_FILE}", header=0, sep=",")
    if prune_technique == "remove-outliers":
        # apply outlier removal using IQR
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]
    elif prune_technique == "remove-duplicates":
        # remove duplicate rows
        data.drop_duplicates(inplace=True)
        data.dropna()

    elif prune_technique == "none":
        # apply no pruning
        pass
    else:
        raise ValueError("Prune technique not recognized.")

    print(f"Pruned data with technique: {prune_technique}.")

    # write data to file
    base_path_pruned = f"./{DATA_FOLDER}/{data_set_name}/{PRUNE_FOLDER}"
    Path(base_path_pruned).mkdir(exist_ok=True)
    data.to_csv(f"{base_path_pruned}/{prune_technique}_{PRUNE_FILE}", index=False)
    print(f"Written pruned data set to file.")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Random Seed Effects prune data!")
    parser.add_argument('--data_set_name', dest='data_set_name', type=str, required=True)
    parser.add_argument('--prune_technique', dest='prune_technique', type=str, required=True)
    args = parser.parse_args()

    print("Pruning data with arguments: ", args.__dict__)
    prune_data(args.data_set_name, args.prune_technique)
