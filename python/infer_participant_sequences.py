import sys
import numpy as np
from learning_utils import pickle_load, pickle_save, get_normalized_features,\
                            get_modified_weights, create_dir
from computational_microscope import ComputationalMicroscope
from analysis_utils import get_data

def modify_clicks(click_sequence):
    modified_clicks = []
    for clicks in click_sequence:
        modified_clicks.append([int(c) for c in clicks] + [0])
    return modified_clicks

# Change this to change which data is loaded
def get_participant_data(exp_num, pid, block=None):
    data = get_data(exp_num)
    clicks_data = data['mouselab-mdp']
    print(block)
    if block:
        clicks_data = clicks_data[(clicks_data.pid == pid) & (clicks_data.block == block)]
    else:
        clicks_data = clicks_data[clicks_data.pid == pid]
    click_sequence = [q['click']['state']['target'] for q in clicks_data.queries]
    click_sequence = modify_clicks(click_sequence)
    envs = [[0]+sr[1:] for sr in clicks_data.stateRewards]
    return click_sequence, envs

def infer_strategies(click_sequences, envs, pipeline, strategy_space,
                    W, features, normalized_features):
    cm = ComputationalMicroscope(pipeline, strategy_space, W, features, normalized_features)
    S, _, _, T = cm.infer_sequences(click_sequences, envs)
    return S, T

if __name__ == "__main__":
    pid = int(sys.argv[1])
    exp_num = sys.argv[2]
    block = None
    if len(sys.argv) > 3:
        block = sys.argv[3]

    strategy_space = pickle_load("data/strategy_space.pkl")
    features = pickle_load("data/microscope_features.pkl")
    strategy_weights = pickle_load("data/microscope_weights.pkl")
    num_features = len(features)
    exp_pipelines = pickle_load("data/exp_pipelines.pkl")
    exp_reward_structures = {'v1.0': 'high_increasing', 'F1': 'high_increasing', 
                            'c1.1': 'low_constant', 'T1.1': 'large_increasing'}
    
    # Defaults for 312 increasing variance task
    reward_structure = "high_increasing"
    pipeline = [exp_pipelines["v1.0"][0]]*100
    
    if exp_num in exp_pipelines:
        reward_structure = exp_reward_structures[exp_num]
        pipeline = exp_pipelines[exp_num]

    normalized_features = get_normalized_features(reward_structure)
    W = get_modified_weights(strategy_space, strategy_weights)

    # TODO:
    # Get clicks and envs of a particular participant
    clicks, envs = get_participant_data(exp_num, pid, block=block)
    S, T = infer_strategies(clicks, envs, pipeline, strategy_space,
                            W, features, normalized_features)

    path = f"results/inferred_sequences/{exp_num}"
    create_dir(path)
    if not block:
        pickle_save(S, f"{path}/{pid}_strategies.pkl")
        pickle_save(T, f"{path}/{pid}_temperature.pkl")
    else:
        pickle_save(S, f"{path}/{pid}_{block}_strategies.pkl")
        pickle_save(T, f"{path}/{pid}_{block}_temperature.pkl")



