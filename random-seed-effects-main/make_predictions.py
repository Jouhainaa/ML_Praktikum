import argparse
from pathlib import Path
import binpickle
import numpy as np
import pandas as pd
import pickle as pkl
from static import *


def make_predictions(data_set_name, prune_technique, split_technique, test_fold, shuffle_seed, MLModel,
                     ML_seed, num_batches, run_batch):
    # get test data
    test_data_path = f"./{DATA_FOLDER}/{data_set_name}/{SPLIT_FOLDER}/" \
                     f"{test_fold}_{shuffle_seed}_{prune_technique}_{split_technique}_{SPLIT_FILE}"
    test_data = pd.read_csv(test_data_path, header=0, sep=",")

    # clean the data
    for column in test_data.columns:
        if test_data[column].dtype == 'object':
            test_data[column] = pd.to_numeric(test_data[column], errors='coerce')
    test_data.dropna(inplace=True)

    users = test_data["X"].unique()
    user_batches = np.array_split(users, num_batches)

    # load ML Model
    base_path_ML = f"./{DATA_FOLDER}/{data_set_name}/{ML_FOLDER}_{MLModel}"
    ML_file_path = f"{base_path_ML}/" \
        f"{test_fold}_{shuffle_seed}_{prune_technique}_{split_technique}_" \
        f"{ML_seed}_{ML_FILE}"
    ML_alg = binpickle.load(ML_file_path)

    X_test = test_data["X"].values.reshape(-1, 1)
    y_test = test_data["y"]

    MLPredictions = {user: ML_alg.predict(X_test) for user in user_batches[run_batch]}

    # save predictions to file
    base_path_predictions = f"./{DATA_FOLDER}/{data_set_name}/{PREDICTION_FOLDER}_{MLModel}"
    Path(base_path_predictions).mkdir(parents=True, exist_ok=True)

    prediction_file_path = f"{base_path_predictions}/" \
                           f"{test_fold}_{shuffle_seed}_{prune_technique}_{split_technique}_" \
                           f"{ML_seed}_{num_batches}_{run_batch}_{PREDICTION_FILE}"
    # Check if the prediction file already exists to avoid overwriting
    if not Path(prediction_file_path).exists():
        with open(prediction_file_path, "wb") as f:
            pkl.dump(MLPredictions, f)
        print(f"Predictions generated and saved: {prediction_file_path}")
    else:
        print(f"Predictions already exist: {prediction_file_path}")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Random Seed Effects make predictions!")
    parser.add_argument('--data_set_name', dest='data_set_name', type=str, required=True)
    parser.add_argument('--prune_technique', dest='prune_technique', type=str, required=True)
    parser.add_argument('--split_technique', dest='split_technique', type=str, required=True)
    parser.add_argument('--test_fold', dest='test_fold', type=int, required=True)
    parser.add_argument('--shuffle_seed', dest='shuffle_seed', type=int, required=True)
    parser.add_argument('--MLModel', dest='MLModel', type=str, required=True)
    parser.add_argument('--ML_seeding', dest='ML_seeding', type=str, required=True)
    parser.add_argument('--num_batches', dest='num_batches', type=int, required=True)
    parser.add_argument('--run_batch', dest='run_batch', type=int, required=True)
    args = parser.parse_args()

    print("Making predictions with arguments: ", args.__dict__)
    make_predictions(args.data_set_name, args.prune_technique, args.split_technique, args.test_fold,
                     args.shuffle_seed, args.MLModel, args.ML_seeding, args.num_batches, args.run_batch)
