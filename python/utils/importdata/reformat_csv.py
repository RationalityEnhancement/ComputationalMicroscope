import pandas as pd
import numpy as np
from pandas.io.json import json_normalize
import json
import itertools

data = pd.read_csv("data/dataclips.csv", sep=';')

# remove unfinished data entries and reset index, which is also the pid
data['endhit'].replace('', np.nan, inplace=False)
data.dropna(subset=['endhit'], inplace=True)
data = data.reset_index(drop=True)
# data.drop(index=0, inplace=True) #drops first row

# save as participant csv
data_participants = data[["bonus", "status", "beginexp"]]
data_participants.to_csv("participants.csv", index=True, index_label="pid")


def flatten(d, sep="_"):
    """
    This function flattens json strings. It checks whether there are concatenated dicts or lists and flattens them.
    Args:
        d: data
        sep: separator to be added in between the flatted data. Example {a: {b: value}} will be flatted into {a_b: value}

    Returns: flattened OrderedDict

    """
    import collections

    obj = collections.OrderedDict()

    def recurse(t, parent_key=""):
        if isinstance(t, list):
            for i in range(len(t)):
                recurse(t[i], parent_key + sep + str(i) if parent_key else str(i))
        elif isinstance(t, dict):
            for k, v in t.items():
                recurse(v, parent_key + sep + k if parent_key else k)
        else:
            obj[parent_key] = t

    recurse(d)
    return obj


# reward_key_list = []
# reward_value_list = []
#
# action_time_key_list = []
# action_time_value_list = []
#
# actions_key_list = []
# actions_value_list = []
# extract information from the raw csv

# mouselab_mdp = pd.DataFrame(
#     columns=["action_time", "actions", "block", "path", "prs", "queries", "reward", "rt", "score", "simulation_node",
#              "state_rewards", "time_elapsed", "trial_index", "trial_time", "trial_type", "pid"])


def format_json(df, col_name, keyword_dict, no_trials):
    participant_dict = {}
    for index, row in df.iterrows():
        data_dict = json.loads(row[col_name])
        data_dict = flatten(data_dict)

        trial_dict = {}
        for trial_id in range(0, no_trials):

            all_rewards_dict = {}
            for keyword_name, keyword_len in keyword_dict.items():
                reward_key_list = []
                reward_value_list = []

                # create dict for all keywords
                for key_reward in data_dict.keys():
                    if str(key_reward).find(keyword_name) != -1:
                        reward_key_list.append(key_reward)
                for key_reward in reward_key_list:
                    reward_value_list.append(data_dict.get(key_reward))

                all_rewards_dict[keyword_name] = reward_value_list[
                                                 (keyword_len * trial_id):(keyword_len * (trial_id + 1))]

            trial_dict[trial_id] = all_rewards_dict
        participant_dict[index] = trial_dict
    return participant_dict


def save_to_df(participant_dict, name_mapping):
    dataframe_list = []
    for participant_id, trial_data in participant_dict.items():
        new_row = {}
        for trial_index, value in trial_data.items():
            new_row["pid"] = participant_id
            new_row["trial_index"] = trial_index
            # print(participant_id)
            # print(trial_index)
            for trial_type, trial_data in value.items():
                new_row[trial_type] = trial_data
            row_data = new_row.copy()
            # print(row_data)
            dataframe_list.append(row_data)
            # mouselab_mdp.append(new_row, ignore_index=True)
    df = pd.DataFrame(dataframe_list)

    # change the name of the dataframe
    df.rename(name_mapping)
    df.to_csv("test.csv")


# load data
data_mouselab = data[["datastring"]]

# here you can set how the columns of the csv will be named. There are some discrepancies between the csv output from postgres and what is required for the Computational Microscope
name_mapping = {"action_time": "actionTimes",
                "actions": "actions",
                "block": "block",
                "path": "path",
                "queries": "queries",
                "rewards": "reward",
                "rt": "rt",
                "score": "score",
                "simulation_mode": "simulationMode",
                "state_rewards": "stateRewards",
                "time_elapsed": "time_elapsed",
                "trial_index": "trial_index",
                "trial_time": "trialTime",
                "trial_type": "trial_type",
                "pid": "pid"}

# here you have to enter the information you want from the csv and the length of the information
keyworddict = {"action_time": 3,
                "actions": 3,
                "block": 1,
                "path": 4,
                "queries": 1,
                "rewards": 3,
                "rt": 3,
                "score": 1,
                "simulation_mode": 3,
                "state_rewards": 13,
                "time_elapsed": 1,
                "trial_index": 1,
                "trial_time": 1,
                "trial_type": 1}

participants_dict = format_json(data_mouselab, "datastring", keyword_dict=keyworddict, no_trials=2)
save_to_df(participants_dict, name_mapping)

# reward_dict = {}
# for index, row in data_mouselab.iterrows():
#     data_dict = json.loads(row["datastring"])
#     data_dict = flatten(data_dict)
#
#     reward_key_list = []
#     reward_value_list = []
#
#     # create dict for rewards
#     for key_reward in data_dict.keys():
#         if str(key_reward).find("rewards") != -1:
#             reward_key_list.append(key_reward)
#     for key_reward in reward_key_list:
#         reward_value_list.append(data_dict.get(key_reward))
#
#     temp = dict(zip(reward_key_list, reward_value_list))
#     reward_dict[index] = temp
#     # print(temp)
#     # print(index)
#     # print(reward_key_list, reward_value_list)
#     # reward_dict.update(dict(zip(itertools.repeat(int(index)), temp)))
#     # reward_dict = dict(zip(reward_key_list, reward_value_list))
# print(reward_dict)

#     # create dict for action times
#     for key_aT in data_dict.keys():
#             if str(key_aT).find("actionTimes") != -1:
#                 action_time_key_list.append(key_aT)
#     for key_aT in action_time_key_list:
#         action_time_value_list.append(data_dict.get(key_aT))
#
#
#     # create dict for action
#     for key_action in data_dict.keys():
#             if str(key_action).find("actions") != -1:
#                 actions_key_list.append(key_action)
#     for key_action in actions_key_list:
#         actions_value_list.append(data_dict.get(key_action))
#
# reward_dict = dict(zip(reward_key_list, reward_value_list))
# action_time_dict = dict(zip(action_time_key_list, action_time_value_list))
# action_dict = dict(zip(actions_key_list, actions_value_list))
#
# print(reward_dict)
# print(action_time_dict)
# print(action_dict)


# save as mouselab csv
# data_mouselab_csv = data[["bonus", "status", "beginexp"]]


# # create empty dataframes

#
# #todo: rewards, etc are not in all trials, only the trials with mouselab mdp. Some are quizes etc
#
#
