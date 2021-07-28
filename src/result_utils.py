import numpy as np
import pandas as pd
import operator
from collections import Counter
from experiment_utils import Experiment
from analysis_utils import get_data
from learning_utils import pickle_load, get_normalized_features
from scipy.stats import ttest_ind
import rpy2.robjects as robjects
from rpy2.robjects import r, pandas2ri
from rpy2.robjects.packages import importr
pandas2ri.activate()
BayesFactor = importr('BayesFactor')

cluster_map = pickle_load("data/kl_cluster_map.pkl")

def bayesfactor(f, s):
    x = pd.DataFrame(f, columns=['a'])
    y = pd.DataFrame(s, columns=['a'])
    robjects.globalenv["x"] = x
    robjects.globalenv["y"] = y
    ttest_output = r('print(t.test(y$a, x$a, alternative="greater"))')
    r('bf = ttestBF(x=x$a, y=y$a, nullInterval = c(0, Inf))')
    r('print(bf[1]/bf[2])')

def get_trajectory(S):
    tr = [S[0]]
    for s in S[1:]:
        if s!= tr[-1]:
            tr.append(s)
    return tuple(tr)

def get_trajectories(exp):
    trs = []
    count = 0
    for pid, S in exp.participant_strategies.items():
        #print(S)
        if S:
            count += 1
        trs.append(get_trajectory(S))
    sorted_counts = sorted(Counter(trs).items(), key=operator.itemgetter(1), reverse=True)
    return trs, sorted_counts

def get_s_trajectories(strategies):
    trs = []
    count = 0
    for pid, S in strategies.items():
        #print(S)
        if S:
            count += 1
        trs.append(get_trajectory(S))
    sorted_counts = sorted(Counter(trs).items(), key=operator.itemgetter(1), reverse=True)
    #print(sorted_counts)
    #print(len(sorted_counts), count)
    return trs, sorted_counts

def get_cluster_trajectories(exp):
    trs = []
    count = 0
    for pid, S in exp.participant_strategies.items():
        #print(S)
        if S:
            count += 1
        cluster_S = [cluster_map[s] for s in S]
        trs.append(get_trajectory(cluster_S))
    sorted_counts = sorted(Counter(trs).items(), key=operator.itemgetter(1), reverse=True)
    #print(sorted_counts)
    #print(len(sorted_counts), count)
    return trs, sorted_counts

def get_clusters_trajectories(strategies):
    trs = []
    count = 0
    for pid, S in strategies.items():
        if S:
            count += 1
        cluster_S = [cluster_map[s] for s in S]
        trs.append(get_trajectory(cluster_S))
    sorted_counts = sorted(Counter(trs).items(), key=operator.itemgetter(1), reverse=True)
    return trs, sorted_counts

def get_lengths(trs):
    lengths = [len(tr[1:-1]) for tr in trs]
    print(np.mean(lengths))
    return lengths

def print_props(counts, p=True):
    S = sum([p[1] for p in counts])
    props = [(p[0], np.round(p[1]/S, 2)) for p in counts]
    if p:
        print("\n", props)
    return props
    
def t_test(s1, s2):
    t, p = ttest_ind(s1, s2)
    df = len(s1) + len(s2) - 2
    print(f"t({df}): {np.round(t, 2)}, p: {np.round(p, 4)}")
    return t, p, df

def get_standard_error(vals):
    std = np.std(vals)
    return np.round(std/np.sqrt(len(vals)), 3)

def get_proportion_se(p, n):
    return np.sqrt(p*(1-p)/n)

def get_confusion(s_true, s_pred):
    unique_true_s = np.unique(s_true + s_pred)
    unique_true_s.sort()
    num_unique = unique_true_s.shape[0]
    s_index = {s:i for i,s in enumerate(unique_true_s)}
    confusion_matrix = np.zeros((num_unique, num_unique))
    for t,p in zip(s_true, s_pred):
        confusion_matrix[s_index[t]][s_index[p]] += 1
    return confusion_matrix

def get_accuracy(confusion_matrix):
    counts = []
    total_count = np.sum(confusion_matrix)
    for k in range(confusion_matrix.shape[0]):
        counts.append(confusion_matrix[k][k])
    return np.round(np.sum(counts)/total_count, 3)
        
def get_confusions(s_true, s_pred, cluster_map):
    strategy_confusion = get_confusion(s_true, s_pred)
    c_true = [cluster_map[s+1] for s in s_true]
    c_pred = [cluster_map[s+1] for s in s_pred]
    cluster_confusion = get_confusion(c_true, c_pred)
    return strategy_confusion, cluster_confusion

def get_proportion_confusion(confusion_matrix, strategy_space): # We need inferred wrt true
    num_categories = confusion_matrix.shape[0]
    proportion_confusion = {}
    for i in range(num_categories):
        num_counts = confusion_matrix[:, i]
        num_counts = num_counts/num_counts.sum()
        proportion_confusion[strategy_space[i]] = {strategy_space[j]:np.round(c,3) for j,c in enumerate(num_counts)}
    return proportion_confusion

def compute_average_confusion_index(strategy_confusion, jeffreys_divergence, strategy_space):
    indexes = []
    modified_jd = jeffreys_divergence[[s-1 for s in strategy_space], :]
    for i in range(strategy_confusion.shape[0]):
        for j in range(strategy_confusion.shape[1]):
            C = strategy_confusion[i][j] 
            if i!=j and C!=0:
                x = np.sort(modified_jd[i]).tolist()
                index = x.index(modified_jd[i][j])
                indexes += [index]*int(C)
    print(np.mean(indexes), np.median(indexes))
