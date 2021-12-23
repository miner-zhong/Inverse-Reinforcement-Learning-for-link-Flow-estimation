import numpy as np

import argparse
from args import irl_args
parser = argparse.ArgumentParser()
irl_args(parser)
args = parser.parse_args()

import util




class RoadNetwork(object):
    """Road network described by states, actions and transitions.
       Also contain information such as feature matrices and observed traffic data

    Attributes:
        nb_state: number of states read from the network property file (= number of links)
        n_states: number of all states (= number of links + number of absorbing states)
        nb_action: number of actions read from the network property file (= number of transition types between connected links)
        n_actions: number of all actions (= number of transition types + an extra action that links destination states to abosrbing states)
        nb_feature: number of feature
        tran_info: transition information
        transition_probability: state-to-state transition probability determined by tran_info
        counts: observed traffic volume data
        trajs_obs: observed trajectories
        trajs_scaled: scaled-up trajectories
        trajs_truth: ground-truth trajectories
        ls_destination: list of destination states
        dest_absorb_info: a dictionary containing pairs of destination states with corresponding absorbing states
        feature_matrix: complete feature matrix
        feature_matrix_1: partial feature matrix (number of columns = dimension of the first state feature vector)
        feature_matrix_2: partial feature matrix (number of columns = dimension of the second state feature vector)
        truth_svf: ground truth state visit frequencies
    """



    def __init__(self, args):
        """Initialize RoadNetwork with input data specified in args.py
        """
        self.nb_state, self.nb_action, self.nb_feature = util.load_network_properties(args.network_property_file)

        self.counts = util.load_traffic_count(args.traffic_count_file)

        self.trajs_obs, self.trajs_scaled, self.trajs_truth = util.load_trajectories(args.trajectory_file)

        self.tran_info = util.load_transition_info(args.transition_info_file)

        self.ls_destination, self.dest_absorb_info = self.get_ls_destination()

        self.feature_matrix, self.feature_matrix_1, self.feature_matrix_2 = util.load_feature_matrix(args.state_feature_file,
                                                                                                     self.nb_state,
                                                                                                     self.nb_feature,
                                                                                                     len(self.counts),
                                                                                                     len(self.ls_destination))
        self.truth_svf = self.get_truth_svf()

        self.n_states = self.nb_state + len(self.ls_destination)

        self.n_actions = self.nb_action + 1

        self.transition_probability = np.array([[[self.get_transition_probability(i, j, k)
                                                  for k in range(self.n_states)]
                                                 for j in range(self.n_actions)]
                                                for i in range(self.n_states)])





    def get_ls_destination(self):
        """Determine destination states and absorbing states from the observed trajectory data

        :return: ls_destination: a list containing all destination states
                 dest_absorb_info: a dictionary containing pairs of destination states with corresponding absorbing states
        """

        ls_destination = []
        for traj in self.trajs_obs:
            if not (traj[-1] in ls_destination):
                ls_destination.append(traj[-1])
        ls_destination.sort()

        dest_absorb_info = {}
        for i in range(len(ls_destination)):
            des = ls_destination[i]
            absorb = self.nb_state + i
            dest_absorb_info[str(des)] = absorb

        return ls_destination, dest_absorb_info








    def get_transition_probability(self, i, j, k):
        """Determine transition probability for each state-action-state pair based on the transition information read from input file

        :param i: starting state index
        :param j: action index
        :param k: ending state index
        :return: probability of transiting from the starting state to the ending state through this action
        """

        ls_absorbing = []
        for key in self.dest_absorb_info.keys():
            ls_absorbing.append(self.dest_absorb_info[key])

        if(i in ls_absorbing):
            next_s = i
        else:
            if (i in self.ls_destination):
                next_s = None
                FLAG = False
                if (j == self.nb_action):
                    next_s = self.dest_absorb_info[str(i)]
                else:
                    info = self.tran_info[str(i)]
                    for pair in info['action_next_state_pairs']:
                        if (j == pair[0]):
                            next_s = pair[1]
                            FLAG = True
                    if not (FLAG):
                        next_s = i
            else:
                next_s = None
                FLAG = False
                if (j == self.nb_action):
                    next_s = i
                else:
                    if (str(i) in list(self.tran_info.keys())):
                        info = self.tran_info[str(i)]
                        for pair in info['action_next_state_pairs']:
                            if (j == pair[0]):
                                next_s = pair[1]
                                FLAG = True
                        if not (FLAG):
                            next_s = i
                    else:
                        next_s = i

        if not (next_s==k):
            return 0
        else:
            return 1






    def get_truth_svf(self):
        """Determine the ground-truth state visitation frequencies from the ground-truth trajectory data

        :return: svf: ground truth state visit frequencies
        """
        result = np.zeros(self.nb_state)
        sum = 0
        for traj in self.trajs_truth:
            for s in traj:
                repr = np.zeros(self.nb_state)
                repr[int(s)] = 1
                result += repr
            sum += 1
        result /= sum
        svf = list(result)

        return svf