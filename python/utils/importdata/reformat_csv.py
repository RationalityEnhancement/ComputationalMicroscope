import pandas as pd
import numpy as np
from pandas.io.json import json_normalize
import json

data = pd.read_csv("data/dataclips.csv", sep=';')

# remove unfinished data entries and reset index, which is also the pid
data['endhit'].replace('', np.nan, inplace=False)
data.dropna(subset=['endhit'], inplace=True)
data = data.reset_index(drop=True)
data.drop(index=0, inplace=True)

# save as participant csv
data_participants = data[["bonus", "status", "beginexp"]]
data_participants.to_csv("participants.csv", index=True, index_label="pid")


def flatten(d,sep="_"):
    import collections

    obj = collections.OrderedDict()

    def recurse(t,parent_key=""):

        if isinstance(t,list):
            for i in range(len(t)):
                recurse(t[i],parent_key + sep + str(i) if parent_key else str(i))
        elif isinstance(t,dict):
            for k,v in t.items():
                recurse(v,parent_key + sep + k if parent_key else k)
        else:
            obj[parent_key] = t

    recurse(d)
    return obj



# save as mouselab csv
data_mouselab = data[["datastring"]]
for index, row in data_mouselab.iterrows():
    test_dict = json.loads(row["datastring"])
    test_dict = flatten(test_dict)
    key_list = []

    for key in test_dict.keys():
        if str(key).find("rewards") != -1:
            key_list.append(key)
    print(key_list)
    for key in key_list:
        print(test_dict.get(key))




#    row["datastring"] = pd.json_normalize(row["datastring"].to_dict())
#    print(row)

# data_mouselab["datastring"] = data_mouselab["datastring"].apply(load_as_json)
#data_mouselab = pd.json_normalize(data_mouselab["datastring"])

#data_mouselab["datastring"] = data_mouselab["datastring"].to_dict()
#print(data_mouselab)
#print(data_mouselab.columns)


#print(data_mouselab.dtypes)


# parse datastring into dict
# datastring = data['datastring'].to_dict()# datastring[0] is the first trial
# #print(datastring)
# #json.loads transforms str into dict IF the str uses double quotes
# for trials in datastring:
#     datastring[trials] = json.loads(datastring[trials])
#
# print(type(datastring))
# print(len(datastring[1]['data']))
# print(datastring[1]['data'][8])
# print(datastring[1]['data'][8]['trialdata'])
# print(datastring[1]['data'][8]['trialdata']['stateRewards'])
#
# # create empty dataframes
# mouselab_mdp = pd.DataFrame(
#    columns=["action_time", "actions", "block", "path", "prs", "queries", "reward", "rt", "score", "simulation_node",
#             "state_rewards", "time_elapsed", "trial_index", "trial_time", "trial_type", "pid"])
#
# #todo: rewards, etc are not in all trials, only the trials with mouselab mdp. Some are quizes etc
#
#
# for rounds in data:
#     try: # does reward exist? if not skip
#
# mouselab_mdp['pid']= datastring.keys()
#
# mouselab_mdp['action_time'] = datastring[1]
# mouselab_mdp['actions'] = datastring[1]
# mouselab_mdp['block'] = datastring[1]
# mouselab_mdp['path'] = datastring[1]
# mouselab_mdp['prs'] = datastring[1]
# mouselab_mdp['reward'] = datastring[1]
# mouselab_mdp['rt'] = datastring[1]
# mouselab_mdp['score'] = datastring[1]
# mouselab_mdp['simulation_node'] = datastring[1]
# mouselab_mdp['state_rewards'] = datastring[1]
# mouselab_mdp['time_elapsed'] = datastring[1]
# mouselab_mdp['trial_index'] = datastring[1]
# mouselab_mdp['trial_time'] = datastring[1]
# mouselab_mdp['trial_type'] = datastring[1]
#

# participants = pd.DataFrame(columns=["bonus", "completed", "pid", "startTime"])

# fillin participants.csv
