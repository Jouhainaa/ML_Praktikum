import json
from pathlib import Path
from select_experiment import file

experiment_settings = json.load(open(f"./experiment_{file}.json"))

seeds = {}
for data_set in experiment_settings["DATA_SET_NAMES"]:
    seeds[data_set] = {}
    data_set_seeds = list(set([file.name.split("_")[1] for file in Path(f"./data/original/{data_set}/split").iterdir()]))
    for seed in data_set_seeds:
        seeds[data_set][seed] = {}
        for MLModel in experiment_settings["MLModels"]:
            seeds[data_set][seed][MLModel] = {}
            for file in Path(f"./data/{data_set}/MLModel_{MLModel}").iterdir():
                if seed in file.name and file.suffix == ".txt":
                    seeds[data_set][seed][MLModel][file.name.split("_")[0]] = next(
                        line.strip() for line in open(file, 'r'))

json.dump(seeds, open(f"project_seeds.txt", "w"))

# seeds = json.loads(open(f"project_seeds.txt", "r").read())
