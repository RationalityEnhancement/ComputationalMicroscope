import sys
from learning_utils import pickle_load, pickle_save, get_normalized_features, \
    get_modified_weights, create_dir
from computational_microscope import ComputationalMicroscope
from experiment_utils import Experiment

if __name__ == "__main__":
    reward_structure = sys.argv[1] #increasing_variance, decreasing_variance
    #reward_structure = "constant_variance"
    block = None
    if len(sys.argv) > 2:
        block = sys.argv[2]

    # Initializations
    strategy_space = pickle_load("data/strategy_space.pkl")
    features = pickle_load("data/microscope_features.pkl")
    strategy_weights = pickle_load("data/microscope_weights.pkl")
    num_features = len(features)
    exp_pipelines = pickle_load("data/exp_pipelines.pkl")  # list of all experiments, e.g. v1.0
    exp_reward_structures = {'increasing_variance': 'high_increasing',
                             'constant_variance': 'low_constant',
                             'decreasing_variance': 'high_decreasing',
                             'transfer_task': 'large_increasing'}

    reward_exps = {"increasing_variance": "v1.0",
                   "decreasing_variance": "c2.1",
                   "constant_variance": "c1.1",
                   "transfer_task": "T1.1"}

    exp_num = reward_exps[reward_structure]  # select experiment number, e.g. v1.0 given entered selection
    if exp_num not in exp_pipelines:
        raise (ValueError, "Reward structure not found.")

    pipeline = exp_pipelines[exp_num]  # select from exp_pipeline the selected v1.0
    # pipeline is a list of len 30, each containing a tuple of 2 {[3, 1, 2], some reward function}
    pipeline = [pipeline[0] for _ in range(100)]  # todo: why range 100?

    normalized_features = get_normalized_features(exp_reward_structures[reward_structure])  # tuple of 2
    W = get_modified_weights(strategy_space, strategy_weights)
    cm = ComputationalMicroscope(pipeline, strategy_space, W, features, normalized_features=normalized_features)
    pids = None
    if exp_num == "c2.1":
        exp = Experiment("c2.1", cm=cm, pids=pids, block=block, variance=2442)
    else:
        exp = Experiment(exp_num, cm=cm, pids=pids, block=block)
    exp.infer_strategies(max_evals=2, show_pids=True)

    save_path = f"results/inferred_strategies/{reward_structure}"
    if block:
        save_path += f"_{block}"
    create_dir(save_path)
    strategies = exp.participant_strategies
    temperatures = exp.participant_temperatures
    pickle_save(strategies, f"{save_path}/strategies.pkl")
    pickle_save(temperatures, f"{save_path}/temperatures.pkl")
