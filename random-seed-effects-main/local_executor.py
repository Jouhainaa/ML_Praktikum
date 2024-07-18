import json
import subprocess
from pathlib import Path
from select_experiment import file, stage
from evaluation_report import evaluation_report
from plot_results import plot_results
from static import *
import os
import sklearn
import sys


def execute_clean_data(data_set_names):
    for data_set_name in data_set_names:
        base_path = f"./{DATA_FOLDER}/{data_set_name}/{CLEAN_FOLDER}/{CLEAN_FILE}"
        if not Path(base_path).exists():
            subprocess.run([sys.executable, "clean_data.py", "--data_set_name", f"{data_set_name}"])


def execute_prune_data(data_set_names, prune_techniques):
    for data_set_name in data_set_names:
        for prune_technique in prune_techniques:
            base_path = f"./{DATA_FOLDER}/{data_set_name}/{PRUNE_FOLDER}/{prune_technique}_{PRUNE_FILE}"
            if not Path(base_path).exists():
                subprocess.run(
                    [sys.executable, "prune_data.py", "--data_set_name", f"{data_set_name}", "--prune_technique",
                     f"{prune_technique}"])


def execute_generate_splits(data_set_names, prune_techniques, split_techniques, num_folds, reproducibility_mode):
    for data_set_name in data_set_names:
        for prune_technique in prune_techniques:
            for split_technique in split_techniques:
                def run_script(reproducibility_seed):
                    subprocess.run(
                        [sys.executable, "generate_splits.py", "--data_set_name", f"{data_set_name}", "--prune_technique",
                         f"{prune_technique}", "--split_technique", f"{split_technique}", "--num_folds",
                         f"{num_folds}", "--reproducibility_seed", f"{reproducibility_seed}"])

                if bool(reproducibility_mode):
                    seeds = json.loads(open(f"project_seeds.txt", "r").read())
                    for shuffle_seed in list(seeds[data_set_name].keys()):
                        run_script(shuffle_seed)
                else:
                    run_script(-1)


def execute_fit_ML(data_set_names, prune_techniques, split_techniques, num_folds, MLModels,
                   ML_seeding, reproducibility_mode):
    for data_set_name in data_set_names:
        for prune_technique in prune_techniques:
            for split_technique in split_techniques:
                shuffle_seeds = []
                for file in Path(f"./{DATA_FOLDER}/{data_set_name}/{SPLIT_FOLDER}").iterdir():
                    _, file_seed, file_prune_technique, file_split_technique, _ = file.name.split(".")[0].split("_")
                    if file_prune_technique == prune_technique and file_split_technique == split_technique:
                        shuffle_seeds.append(file_seed)
                shuffle_seeds = list(set(shuffle_seeds))
                for MLModel in MLModels:
                    for shuffle_seed in shuffle_seeds:
                        for ML_seed in ML_seeding:
                            for test_fold in range(num_folds):
                                def run_script(reproducibility_seed):
                                    base_path = f"./{DATA_FOLDER}/{data_set_name}/{ML_FOLDER}_{MLModel}/" \
                                                f"{test_fold}_{shuffle_seed}_{prune_technique}_{split_technique}_" \
                                                f"{ML_seed}_{ML_FILE}"
                                    if not Path(base_path).exists():
                                        subprocess.run(
                                            [sys.executable, "fit_ML.py", "--data_set_name",
                                             f"{data_set_name}", "--prune_technique", f"{prune_technique}",
                                             "--split_technique", f"{split_technique}", "--num_folds", f"{num_folds}",
                                             "--test_fold", f"{test_fold}", "--shuffle_seed", f"{shuffle_seed}",
                                             "--MLModel", f"{MLModel}", "--ML_seeding",
                                             f"{ML_seed}", "--reproducibility_seed",
                                             f"{reproducibility_seed}"])

                                if bool(reproducibility_mode):
                                    seeds = json.loads(open(f"project_seeds.txt", "r").read())
                                    try:
                                        run_script(seeds[data_set_name][shuffle_seed][MLModel][str(test_fold)])
                                    except KeyError:
                                        print(f"Key not found for data_set_name: {data_set_name}, shuffle_seed: {shuffle_seed}, MLModel: {MLModel}, test_fold: {test_fold}")
                                else:
                                    run_script(-1)


def execute_make_predictions(data_set_names, prune_techniques, split_techniques, num_folds, MLModels,
                             ML_seeding, num_batches):
    for data_set_name in data_set_names:
        for prune_technique in prune_techniques:
            for split_technique in split_techniques:
                shuffle_seeds = []
                for file in Path(f"./{DATA_FOLDER}/{data_set_name}/{SPLIT_FOLDER}").iterdir():
                    _, file_seed, file_prune_technique, file_split_technique, _ = file.name.split(".")[0].split("_")
                    if file_prune_technique == prune_technique and file_split_technique == split_technique:
                        shuffle_seeds.append(file_seed)
                shuffle_seeds = list(set(shuffle_seeds))
                for MLModel in MLModels:
                    for shuffle_seed in shuffle_seeds:
                        for ML_seed in ML_seeding:
                            for test_fold in range(num_folds):
                                for run_batch in range(num_batches):
                                    base_path = f"./{DATA_FOLDER}/{data_set_name}/{PREDICTION_FOLDER}_{MLModel}/" \
                                                f"{test_fold}_{shuffle_seed}_{prune_technique}_{split_technique}_" \
                                                f"{ML_seed}_{num_batches}_{run_batch}_{PREDICTION_FILE}"
                                    if not Path(base_path).exists():
                                        subprocess.run(
                                            [sys.executable, "make_predictions.py", "--data_set_name",
                                             f"{data_set_name}", "--prune_technique", f"{prune_technique}",
                                             "--split_technique", f"{split_technique}", "--test_fold", f"{test_fold}",
                                             "--shuffle_seed", f"{shuffle_seed}", "--MLModel", f"{MLModel}",
                                             "--ML_seeding", f"{ML_seed}", "--num_batches",
                                             f"{num_batches}", "--run_batch", f"{run_batch}"])


def execute_evaluate_predictions(data_set_names, prune_techniques, split_techniques, num_folds, MLModels,
                                 ML_seeding, num_batches, topn_scores):
    topn_scores_string = '-'.join([str(x) for x in topn_scores])
    for data_set_name in data_set_names:
        for prune_technique in prune_techniques:
            for split_technique in split_techniques:
                shuffle_seeds = []
                for file in Path(f"./{DATA_FOLDER}/{data_set_name}/{SPLIT_FOLDER}").iterdir():
                    _, file_seed, file_prune_technique, file_split_technique, _ = file.name.split(".")[0].split("_")
                    if file_prune_technique == prune_technique and file_split_technique == split_technique:
                        shuffle_seeds.append(file_seed)
                shuffle_seeds = list(set(shuffle_seeds))
                for MLModel in MLModels:
                    for shuffle_seed in shuffle_seeds:
                        for ML_seed in ML_seeding:
                            for test_fold in range(num_folds):
                                base_path = f"./{DATA_FOLDER}/{data_set_name}/{EVALUATION_FOLDER}_{MLModel}/" \
                                            f"{test_fold}_{shuffle_seed}_{prune_technique}_{split_technique}_" \
                                            f"{ML_seed}_{num_batches}_{topn_scores_string}_" \
                                            f"{EVALUATION_FILE}"
                                if not Path(base_path).exists():
                                    subprocess.run(
                                        [sys.executable, "evaluate_predictions.py", "--data_set_name",
                                         f"{data_set_name}", "--prune_technique", f"{prune_technique}",
                                         "--split_technique", f"{split_technique}", "--test_fold",
                                         f"{test_fold}", "--shuffle_seed", f"{shuffle_seed}", "--MLModel",
                                         f"{MLModel}", "--ML_seeding", f"{ML_seed}",
                                         "--num_batches", f"{num_batches}", "--topn_scores"] + [f"{r}" for r in
                                                                                                topn_scores])


def execute_evaluation_report(data_set_names, prune_techniques, split_techniques, num_folds, MLModels,
                              ML_seeding, num_batches, topn_scores):
    evaluation_report(data_set_names, prune_techniques, split_techniques, num_folds, MLModels,
                      ML_seeding, num_batches, topn_scores)


def execute_plot_results():
    plot_results()


experiment_settings = json.load(open(f"./experiment_full.json"))
if stage == 0:
    execute_clean_data(experiment_settings["DATA_SET_NAMES"])
elif stage == 1:
    execute_prune_data(experiment_settings["DATA_SET_NAMES"], experiment_settings["PRUNE_TECHNIQUES"])
elif stage == 2:
    execute_generate_splits(experiment_settings["DATA_SET_NAMES"], experiment_settings["PRUNE_TECHNIQUES"],
                            experiment_settings["SPLIT_TECHNIQUES"], experiment_settings["NUM_FOLDS"],
                            experiment_settings["REPRODUCIBILITY_MODE"])
elif stage == 3:
    execute_fit_ML(experiment_settings["DATA_SET_NAMES"], experiment_settings["PRUNE_TECHNIQUES"],
                   experiment_settings["SPLIT_TECHNIQUES"], experiment_settings["NUM_FOLDS"],
                   experiment_settings["MLMODELS"], experiment_settings["ML_SEEDING"],
                   experiment_settings["REPRODUCIBILITY_MODE"])
elif stage == 4:
    execute_make_predictions(experiment_settings["DATA_SET_NAMES"], experiment_settings["PRUNE_TECHNIQUES"],
                             experiment_settings["SPLIT_TECHNIQUES"], experiment_settings["NUM_FOLDS"],
                             experiment_settings["MLMODELS"], experiment_settings["ML_SEEDING"],
                             experiment_settings["NUM_BATCHES"])
elif stage == 5:
    execute_evaluate_predictions(experiment_settings["DATA_SET_NAMES"], experiment_settings["PRUNE_TECHNIQUES"],
                                 experiment_settings["SPLIT_TECHNIQUES"], experiment_settings["NUM_FOLDS"],
                                 experiment_settings["MLMODELS"], experiment_settings["ML_SEEDING"],
                                 experiment_settings["NUM_BATCHES"], experiment_settings["TOPN_SCORES"])
elif stage == 6:
    execute_evaluation_report(experiment_settings["DATA_SET_NAMES"], experiment_settings["PRUNE_TECHNIQUES"],
                              experiment_settings["SPLIT_TECHNIQUES"], experiment_settings["NUM_FOLDS"],
                              experiment_settings["MLMODELS"], experiment_settings["ML_SEEDING"],
                              experiment_settings["NUM_BATCHES"], experiment_settings["TOPN_SCORES"])
elif stage == 7:
    execute_plot_results()

else:
    print("No valid stage selected!")
