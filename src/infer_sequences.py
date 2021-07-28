import sys
import numpy as np
from analysis_utils import get_data
from learning_utils import pickle_load, pickle_save, get_normalized_features,\
                            get_modified_weights, create_dir
from computational_microscope import ComputationalMicroscope
from experiment_utils import Experiment

def print_help():
    help_str = """
        This script takes three parameters (<dataset>, <block>, <condition>).
        1. <dataset> parameter defines which dataset to run the experiment on.
            The values it can take out of the box are: increasing_variance,
            constant_variance, decreasing_variance and transfer_task. In general,
            it determines the reward structure the computational microscope is run on.
        2. <block> parameter defines which subset of the trials the method is run on.
            The values it can take are train and test for the increasing_variance environment
            and test for all the other three environments. In general, it can take any value
            defined in the block column of the mouselab-mdp.csv file of the dataset. In addition,
            the parameter value can be none which would run the method on all the available trials.
        3. <condition> parameter defines which subset of trials to run the method on. In general, it
            can take any value defined in the condition column of the participants.csv file of the
            dataset.
    """
    print(help_str)

if __name__ == "__main__":
    reward_structure = sys.argv[1]
    block = None

    if reward_structure == "help":
        print_help()
        exit()

    if len(sys.argv) > 2:
        block = sys.argv[2]

    # Initializations
    strategy_space = pickle_load("data/strategy_space.pkl")
    features = pickle_load("data/microscope_features.pkl")
    strategy_weights = pickle_load("data/microscope_weights.pkl")
    num_features = len(features)
    exp_pipelines = pickle_load("data/exp_pipelines.pkl")
    exp_reward_structures = {'increasing_variance': 'high_increasing', 
                            'constant_variance': 'low_constant',
                            'decreasing_variance': 'high_decreasing',
                            'transfer_task': 'large_increasing'}
    
    reward_exps = {"increasing_variance": "v1.0",
                  "decreasing_variance": "c2.1_dec",
                  "constant_variance": "c1.1",
                  "transfer_task": "T1.1"}

    exp_num = reward_exps[reward_structure]
    if exp_num not in exp_pipelines:
        raise(ValueError, "Reward structure not found.")
    
    pipeline = exp_pipelines[exp_num]
    pipeline = [pipeline[0] for _ in range(100)]

    normalized_features = get_normalized_features(exp_reward_structures[reward_structure])
    W = get_modified_weights(strategy_space, strategy_weights)
    cm = ComputationalMicroscope(pipeline, strategy_space, W, features, normalized_features=normalized_features)
    pids = None
    if exp_num == "c2.1_dec":
        exp = Experiment("c2.1", cm=cm, pids=pids, block = block, variance = 2442)
    else:
        exp = Experiment(exp_num, cm=cm, pids=pids, block = block)
    exp.infer_strategies(max_evals=2, show_pids=True)

    save_path = f"results/inferred_strategies/{reward_structure}"
    if block:
        save_path += f"_{block}"
    create_dir(save_path)
    strategies = exp.participant_strategies
    temperatures = exp.participant_temperatures
    pickle_save(strategies, f"{save_path}/strategies.pkl")
    pickle_save(temperatures, f"{save_path}/temperatures.pkl")