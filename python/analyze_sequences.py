import sys
from python.utils.learning_utils import pickle_load, get_normalized_features,\
                            get_modified_weights
from computational_microscope import ComputationalMicroscope
from python.utils.utils import Experiment

if __name__ == "__main__":
    reward_structure = sys.argv[1]
    block = None
    if len(sys.argv) > 2:
        block = sys.argv[2]

    # Initializations
    strategy_space = pickle_load("data/strategy_space.pkl")
    features = pickle_load("data/microscope_features.pkl")
    strategy_weights = pickle_load("data/microscope_weights.pkl")
    num_features = len(features)
    exp_pipelines = pickle_load("data/exp_pipelines.pkl")
    features = pickle_load("data/microscope_features.pkl")
    decision_systems = pickle_load("data/decision_systems.pkl")
    feature_systems = pickle_load("data/feature_systems.pkl")
    decision_system_features = pickle_load("data/decision_system_features.pkl")
    DS_proportions = pickle_load("data/strategy_decision_proportions.pkl")
    W_DS = pickle_load("data/strategy_decision_weights.pkl")
    cluster_map = pickle_load("data/kl_cluster_map.pkl")
    strategy_scores = pickle_load("data/strategy_scores.pkl")
    cluster_scores = pickle_load("data/cluster_scores.pkl")

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

    dir_path = f"results/inferred_strategies/{reward_structure}"
    if block:
        dir_path += f"_{block}"
    
    try:
        strategies = pickle_load(f"{dir_path}/strategies.pkl")
        temperatures = pickle_load(f"{dir_path}/temperatures.pkl")
    except Exception as e:
        print(e)
        exit()

    exp.summarize(features, normalized_features, strategy_weights, 
                decision_systems, W_DS, DS_proportions, strategy_scores, 
                cluster_scores, cluster_map, precomputed_strategies=strategies,
                precomputed_temperatures=temperatures,
                show_pids=False)
