import csv
import numpy as np


def load_network_properties(path):
    """Load network properties from input file

    :param path: data file path

    :return: nb_state: number of states in the road network (=the number of links), excluding the number of absorbing states
             nb_action: number of actions in the road network (= the number of transition types between connected links)
             nb_feature: dimension of the first state feature vector, which is used to convey distribution information from observed trajectories.
                         In Nguyen-Dupuis network, feature vectors are defined using unique state id, nb_feature = 38
                         In Berlin network,
                            - if feature vectors are defined using unique state id, nb_feature = 532
                            - if feature vectors are defined using road characteristics, nb_feature = 92
    """

    with open(path,'r') as csvfile:
        reader = csv.reader(csvfile)
        rows= [row for row in reader]

    idx_3 = rows[0].index('nb_state')
    nb_state = int(rows[1][idx_3])

    idx_4 = rows[0].index('nb_action')
    nb_action = int(rows[1][idx_4])

    idx_5 = rows[0].index('nb_feature')
    nb_feature = int(rows[1][idx_5])

    return nb_state, nb_action, nb_feature






def load_traffic_count(path):
    """Load observed traffic volume data from input file

    :param path: data file path

    :return: counts: a dictionary containing observed volume data for each observed link
    """

    with open(path,'r') as csvfile:
        reader = csv.reader(csvfile)
        rows= [row for row in reader]

    counts = {}
    ls_state, ls_nb = [], []

    idx_1 = rows[0].index('obs_state')
    for i in range(1, len(rows)):
        ls_state.append(rows[i][idx_1])

    idx_2 = rows[0].index('observed_nb')
    for i in range(1, len(rows)):
        ls_nb.append(rows[i][idx_2])

    for i in range(len(ls_nb)):
        counts[str(ls_state[i])] = int(ls_nb[i])

    return counts








def load_trajectories(path):
    """Load observed trajectories data from input file

    :param path: data file path

    :return: obs_trajs: arrays of all observed trajectories
             obs_trajs_scaled: arrays of all observed trajectories scaled up using factors determined by running the optimization model,
                               which is designed to find an approximated population trajetcory number.
             obs_trajs_truth: arrays of ground-truth population trajectories
    """


    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]

    ls_traj, ls_nb, ls_nb_SCALED, ls_nb_truth = [], [], [], []

    for r in rows[1:]:
        idx_1 = rows[0].index('ls_states')
        ls_traj.append(r[idx_1])
        idx_2 = rows[0].index('obs_flow')
        ls_nb.append(r[idx_2])
        idx_3 = rows[0].index('obs_flow_scaled')
        ls_nb_SCALED.append(r[idx_3])
        idx_4 = rows[0].index('ground_truth')
        ls_nb_truth.append(r[idx_4])

    trajectories_obs, trajectories_scaled, trajectories_truth = [], [], []

    for i in range(len(ls_traj)):
        traj = ls_traj[i]
        t = []
        if (',' in traj):
            info = traj.split(',')
            t.append(int(info[0][1:]))
            for k in range(1, len(info) - 1):
                t.append(int(info[k]))
            t.append(int(info[-1][:-1]))
        else:
            t.append(int(traj[1:-1]))

        for a in range(int(ls_nb[i])):
            trajectories_obs.append(t)

        for b in range(int(ls_nb_SCALED[i])):
            trajectories_scaled.append(t)

        for c in range(int(ls_nb_truth[i])):
            trajectories_truth.append(t)

    obs_trajs = np.array(trajectories_obs)
    obs_trajs_scaled = np.array(trajectories_scaled)
    obs_trajs_truth = np.array(trajectories_truth)

    return obs_trajs, obs_trajs_scaled, obs_trajs_truth





def load_transition_info(path):
    """Load state-to-state transition information from input file

    :param path: data file path

    :return: tran_info: a dictionary containing transition information between any pair of connected states
    """

    with open(path,'r') as csvfile:
        reader = csv.reader(csvfile)
        rows= [row for row in reader]

    tran_info = {}
    idx_1 = rows[0].index('state_from')
    idx_2 = rows[0].index('state_to')
    idx_3 = rows[0].index('action')
    for r in rows[1:]:
        if not (str(r[idx_1]) in tran_info.keys()):
            tran_info[str(r[idx_1])] = {'action_next_state_pairs': [[int(r[idx_3]), int(r[idx_2])]]}
        else:
            tran_info[str(r[idx_1])]['action_next_state_pairs'].append([int(r[idx_3]), int(r[idx_2])])

    return tran_info







def load_feature_matrix(path, nb_state, nb_feature, nb_obs_state, nb_destination):
    """Load feature matrix from input file

    number of rows = number of all states
                   = number of links + number of absorbing states
                   = number of links + number of destination states

    number of columns = dimension of the first state feature vector + dimension of the second state feature vector
                      = nb_state read from the network property file + number of observed states

    :param path: data file path
    :param nb_state: number of links read from the network property file
    :param nb_feature: dimension of the first state feature vector from the network property file
    :param nb_obs_state: number of observed states obtained from the observed traffic volume data
    :param nb_destination: number of destinations obtained from the observed trajectory data

    :return: feature_all: the conplete feature matrix
             feature_e1: feature matrix only containing columns corresponding to the first state feature vector
             feature_e2: feature matrix only containing columns corresponding to the second state feature vector
    """


    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]

    ls_all, ls_e1, ls_e2 = [], [], []

    for a in range(nb_feature):
        feature = 'f' + str(a)
        idx = rows[0].index(feature)
        info = []
        for r in rows[1:]:
            info.append(float(r[idx]))
        ls_all.append(info)
        ls_e1.append(info)

    for j in range(nb_obs_state):
        idx = rows[0].index('L' + str(j))
        info = []
        for r in rows[1:]:
            info.append(float(r[idx]))
        ls_all.append(info)
        ls_e2.append(info)

    f_all, f_e1, f_e2 = [], [], []

    for s in range(nb_state):
        f0 = []
        for a in range(len(ls_all)):
            f0.append(ls_all[a][s])
        f_all.append(f0)

        f1 = []
        for b in range(len(ls_e1)):
            f1.append(ls_e1[b][s])
        f_e1.append(f1)

        f2 = []
        for c in range(len(ls_e2)):
            f2.append(ls_e2[c][s])
        f_e2.append(f2)

    # for all absorbing states, the state feature vectors contain all zeros

    for s in range(nb_destination):
        f0 = []
        for a in range(len(ls_all)):
            f0.append(0)
        f_all.append(f0)

        f1 = []
        for b in range(len(ls_e1)):
            f1.append(0)
        f_e1.append(f1)

        f2 = []
        for c in range(len(ls_e2)):
            f2.append(0)
        f_e2.append(f2)

    feature_all = np.array(f_all)
    feature_e1 = np.array(f_e1)
    feature_e2 = np.array(f_e2)

    return feature_all, feature_e1, feature_e2



