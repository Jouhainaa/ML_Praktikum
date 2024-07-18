import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd
import pickle as pkl

from static import *


def evaluate_predictions(data_set_name, prune_technique, split_technique, test_fold, shuffle_seed, MLModel,
                         ML_seed, num_batches, topn_scores):
    # get test data
    test_data = pd.read_csv(
        f"./{DATA_FOLDER}/{data_set_name}/{SPLIT_FOLDER}/"
        f"{test_fold}_{shuffle_seed}_{prune_technique}_{split_technique}_{SPLIT_FILE}", header=0, sep=",")

    evaluation_data = {}
    for run_batch in range(num_batches):
        # get predictions
        predictions = pkl.load(open(
            f"./{DATA_FOLDER}/{data_set_name}/"
            f"{PREDICTION_FOLDER}_{MLModel}/{test_fold}_{shuffle_seed}_{prune_technique}_{split_technique}_"
            f"{ML_seed}_{num_batches}_{run_batch}_{PREDICTION_FILE}", "rb"))
        if run_batch not in evaluation_data:
            evaluation_data[run_batch] = {}

        for topn_score in topn_scores:
            topn_score = int(topn_score)
            # calculate MAE and MSE
            mae_per_user = []
            mse_per_user = []
            for user, user_predictions in predictions.items():
                if user_predictions.shape[0] < topn_score:
                    mae_per_user.append(0)
                    mse_per_user.append(0)
                    continue
                top_k_predictions = np.array(user_predictions[:topn_score], dtype=float)
                positive_test_interactions = test_data["X"] == user
                positive_test_interactions = np.where(positive_test_interactions == 'No Data', np.nan, positive_test_interactions).astype(float)
                positive_test_interactions = np.nan_to_num(positive_test_interactions, nan=0.0)
                if len(positive_test_interactions) == 0:
                    continue
                true_values = np.array(positive_test_interactions[:topn_score], dtype=float)
                predicted_values = top_k_predictions[:topn_score]
                user_mae = np.mean(np.abs(predicted_values - true_values))
                user_mse = np.mean((predicted_values - true_values) ** 2)
                print(user_mse)
                mae_per_user.append(user_mae)
                mse_per_user.append(user_mse)
            total_mae = np.mean(mae_per_user)
            total_rmse = math.sqrt(np.mean(mse_per_user))
            evaluation_data[run_batch][topn_score] = {"mae": total_mae, "rmse": total_rmse}

    # save results
    base_path_evaluations = f"./{DATA_FOLDER}/{data_set_name}/{EVALUATION_FOLDER}_{MLModel}"
    Path(base_path_evaluations).mkdir(parents=True, exist_ok=True)
    topn_scores_string = '-'.join([str(x) for x in topn_scores])
    with open(f"{base_path_evaluations}/"
              f"{test_fold}_{shuffle_seed}_{prune_technique}_{split_technique}_"
              f"{ML_seed}_{num_batches}_{topn_scores_string}_{EVALUATION_FILE}", "wb") as f:
        pkl.dump(evaluation_data, f)
    print(f"Evaluated predictions and saved results.")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Random Seed Effects evaluate predictions!")
    parser.add_argument('--data_set_name', dest='data_set_name', type=str, required=True)
    parser.add_argument('--prune_technique', dest='prune_technique', type=str, required=True)
    parser.add_argument('--split_technique', dest='split_technique', type=str, required=True)
    parser.add_argument('--test_fold', dest='test_fold', type=int, required=True)
    parser.add_argument('--shuffle_seed', dest='shuffle_seed', type=int, required=True)
    parser.add_argument('--MLModel', dest='MLModel', type=str, required=True)
    parser.add_argument('--ML_seeding', dest='ML_seeding', type=str, required=True)
    parser.add_argument('--num_batches', dest='num_batches', type=int, required=True)
    parser.add_argument('--topn_scores', dest='topn_scores', nargs="+", type=str, required=True)
    args = parser.parse_args()

    print("Evaluating predictions with arguments: ", args.__dict__)
    evaluate_predictions(args.data_set_name, args.prune_technique, args.split_technique, args.test_fold,
                         args.shuffle_seed, args.MLModel, args.ML_seeding, args.num_batches,
                         args.topn_scores)
