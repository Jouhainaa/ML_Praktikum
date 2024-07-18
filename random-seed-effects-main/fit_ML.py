import argparse
from pathlib import Path
import binpickle
import numpy as np
import pandas as pd
from static import *
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
import joblib


def fit_ML(data_set_name, prune_technique, split_technique, num_folds, test_fold, shuffle_seed, MLModel,
           ML_seed, reproducibility_seed):
    # get train data
    train_folds = [x for x in range(num_folds) if x != test_fold]
    train_data_dfs = []
    for train_fold in train_folds:
        train_data_dfs.append(
            pd.read_csv(f"./{DATA_FOLDER}/{data_set_name}/{SPLIT_FOLDER}/"
                        f"{train_fold}_{shuffle_seed}_{prune_technique}_{split_technique}_{SPLIT_FILE}",
                        header=0, sep=",", skiprows=0))
    train_data = pd.concat(train_data_dfs, ignore_index=True)
    for column in train_data.columns:
        if train_data[column].dtype == 'object':
            print(f"Column '{column}' contains non-numeric data. Attempting to clean it.")
            train_data[column] = pd.to_numeric(train_data[column], errors='coerce')

    # Drop rows with NaN values that were created by coercing non-numeric data
    train_data.dropna(inplace=True)
    print(train_data.head())
    # obtain seed for MLModel
    if ML_seed == "random":
        if reproducibility_seed == -1:
            ML_seed_actual = np.random.randint(0, np.iinfo(np.int32).max)
        else:
            ML_seed_actual = reproducibility_seed
    elif ML_seed == "static":
        ML_seed_actual = 42
    else:
        raise ValueError("ML seeding method not recognized.")

    # select the recommender
    if MLModel == "decision_tree":
        ML_alg = DecisionTreeRegressor(random_state=ML_seed_actual)
    elif MLModel == "knn":
        ML_alg = KNeighborsRegressor()
    elif MLModel == "linear_regression":
        ML_alg = LinearRegression()
    else:
        raise ValueError("ML Model not supported!")
    # todo
    if 'X' in train_data.columns and 'y' in train_data.columns:
        X = train_data['X'].values.reshape(-1, 1)
        y = train_data['y']
    else:
        raise ValueError("Required columns 'x' and 'y' are not in the dataset.")

    # fit machine learning model
    ML_alg.fit(X, y)

    # save machine learning model to file
    base_path_ML = f"./{DATA_FOLDER}/{data_set_name}/{ML_FOLDER}_{MLModel}"
    Path(base_path_ML).mkdir(exist_ok=True)
    binpickle.dump(ML_alg, f"{base_path_ML}/"
                   f"{test_fold}_{shuffle_seed}_{prune_technique}_{split_technique}_"
                   f"{ML_seed}_{ML_FILE}")
    with open(f"{base_path_ML}/"
              f"{test_fold}_{shuffle_seed}_{prune_technique}_{split_technique}_"
              f"{ML_seed}_{ML_SEED_FILE}", "w") as f:
        f.write(f"{ML_seed_actual}")
    print(f"Fitted ML Model and saved to file.")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Random Seed Effects fit machine learning model!")
    parser.add_argument('--data_set_name', dest='data_set_name', type=str, required=True)
    parser.add_argument('--prune_technique', dest='prune_technique', type=str, required=True)
    parser.add_argument('--split_technique', dest='split_technique', type=str, required=True)
    parser.add_argument('--num_folds', dest='num_folds', type=int, required=True)
    parser.add_argument('--test_fold', dest='test_fold', type=int, required=True)
    parser.add_argument('--shuffle_seed', dest='shuffle_seed', type=int, required=True)
    parser.add_argument('--MLModel', dest='MLModel', type=str, required=True)
    parser.add_argument('--ML_seeding', dest='ML_seeding', type=str, required=True)
    parser.add_argument('--reproducibility_seed', dest='reproducibility_seed', type=int, required=True)
    args = parser.parse_args()

    print("Fitting Machine learning Model with arguments: ", args.__dict__)
    fit_ML(args.data_set_name, args.prune_technique, args.split_technique, args.num_folds, args.test_fold,
           args.shuffle_seed, args.MLModel, args.ML_seeding, args.reproducibility_seed)
