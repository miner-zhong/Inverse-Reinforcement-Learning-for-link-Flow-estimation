import numpy as np

import util

from DiscreteEnv import DiscreteEnv




class RoadEnv(DiscreteEnv):

    """Road network described by states, actions and transitions.
       Also contain information such as feature matrices and observed traffic data

    Attributes:
        nb_state: number of states read from the network property file (= number of links)
        nS: number of all states (= number of links + number of absorbing states)
        nb_action: number of actions read from the network property file (= number of transition types between connected links)
        nA: number of all actions (= number of transition types + an extra action that links destination states to abosrbing states)
        nb_feature: number of feature
        tran_info: transition information
        transition: a dictionary containing transition probability of each state-action-state
        counts: observed traffic volume data
        trajs_obs: observed trajectories
        trajs_scaled: scaled-up trajectories
        trajs_truth: ground-truth trajectories
        ls_origin: list of origin states
        ls_destination: list of destination states
        ls_absorbing: list of absorbing states
        dest_absorb_info: a dictionary containing pairs of destination states with corresponding absorbing states
        fm_1: partial feature matrix (number of columns = dimension of the first state feature vector)
        fm_2: partial feature matrix (number of columns = dimension of the second state feature vector)
        expert_fe_1: the first expert feature expectation (based on observed trajectory data)
        expert_fe_2: the second expert feature expectation (based on observed traffic volume data)
        truth_svf: ground truth state visit frequencies
    """



    def __init__(self, args):
        """Initialize RoadEnv with input data specified in args.py
        """

        self.nb_state, self.nb_action, self.nb_feature = util.load_network_properties(args.network_property_file)

        self.counts = util.load_traffic_count(args.traffic_count_file)

        self.trajs_obs, self.trajs_scaled, self.trajs_truth = util.load_trajectories(args.trajectory_file)

        self.ls_origin = self.get_ls_origin()

        self.ls_destination, self.dest_absorb_info, self.ls_absorbing = self.get_ls_destination()

        self.nS = self.nb_state + len(self.ls_destination)

        self.nA = self.nb_action + 1

        self.tran_info = util.get_transition_info(args.transition_info_file)
        self.transition = self.get_transition()

        self.fm_1, self.fm_2 = util.load_feature_matrix(args.state_feature_file,
                                                        self.nb_state, self.nb_feature,
                                                        len(self.counts), len(self.ls_destination))
        self.expert_fe_1 = self.get_expert_1()

        self.expert_fe_2 = self.get_expert_2()

        self.truth_svf = self.get_truth_svf()

        super(RoadEnv, self).__init__(self.nS, self.nA, self.transition, self.get_isd())




    def state_from_repr_to_idx(self, repr):
        """Transform one-hot vector into state index

        :param repr: one-hot vector of the state
        :return: index of the state
        """
        location = None
        size = len(repr)
        for i in range(size):
            if(repr[i]==1.0):
                location = i
        return location




    def step(self, action):
        """move to next state given current state the chosen action

        :param action: index of the chosen action
        :return: state: index of the next state
                 done: = True if the next state is absorbing state, = False otherwise
        """
        state_idx, env_reward, done, info = DiscreteEnv.step(self, action)
        vec = np.zeros(self.nS)
        vec[state_idx] = 1.0
        state = np.array(vec)
        return state, done




    def reset(self):
        """reset the environment

        :return: a chosen starting state for the agent
        """
        state_idx = DiscreteEnv.reset(self)
        vec = np.zeros(self.nS)
        vec[state_idx] = 1.0
        state = np.array(vec)
        return state




    def get_transition(self):
        """determine the transition probability for each state-action-action pair

        :return: P: the transition information P[state][action] == [(probability, nextstate, reward, done)]
        """

        P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}

        for s in range(self.nS):
            for a in range(self.nA):
                li = P[s][a]

                if (s in self.ls_absorbing):
                    next_s = s
                    li.append((1.0, next_s, 0, True))
                else:
                    if (s in self.ls_destination):
                        next_s = None
                        FLAG = False
                        if (a == self.nb_action):
                            next_s = self.dest_absorb_info[str(s)]
                        else:
                            info = self.tran_info[str(s)]
                            for pair in info['action_next_state_pairs']:
                                if (a == pair[0]):
                                    next_s = pair[1]
                                    FLAG = True
                            if not (FLAG):
                                next_s = s
                        li.append((1.0, next_s, -1, False))
                    else:
                        next_s = None
                        FLAG = False
                        if (a == self.nb_action):
                            next_s = s
                        else:
                            if (str(s) in list(self.tran_info.keys())):
                                info = self.tran_info[str(s)]
                                for pair in info['action_next_state_pairs']:
                                    if (a == pair[0]):
                                        next_s = pair[1]
                                        FLAG = True
                                if not (FLAG):
                                    next_s = s
                            else:
                                next_s = s
                        li.append((1.0, next_s, -1, False))

        return P





    def get_ls_origin(self):
        """Determine origin states from the observed trajectory data

        :return: ls_origin: a list containing all origin states
        """
        ls_origin = []
        for traj in self.trajs_obs:
            if not (traj[0] in ls_origin):
                ls_origin.append(traj[0])
        ls_origin.sort()
        return ls_origin




    def get_ls_destination(self):
        """Determine destination states and absorbing states from the observed trajectory data

        :return: ls_destination: a list containing all destination states
                 ls_absorbing: a list containing all absorbing states
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

        ls_absorbing = []
        for key in dest_absorb_info.keys():
            ls_absorbing.append(dest_absorb_info[key])

        return ls_destination, dest_absorb_info, ls_absorbing





    def get_isd(self):
        """Determine initial state distribution from the observed trajectory data

        :return: isd: a list containing probability of each state being the starting state
        """

        isd_info = {}
        for o in self.ls_origin:
            isd_info[str(o)] = 0

        sum = 0
        for traj in self.trajs_obs:
            isd_info[str(traj[0])] += 1
            sum += 1

        for k in isd_info.keys():
            isd_info[k] /= sum

        isd = []
        for s in range(self.nS):
            isd.append(0)
        for origin in self.ls_origin:
            isd[origin] = isd_info[str(origin)]
        isd = np.array(isd)

        return isd





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





    def get_expert_1(self):
        """Determine the first expert feature expectation based on observed trajectory data

        :return: expert_fe_1: the first expert feature expectation
        """

        freq = np.zeros(self.nS)

        for traj in self.trajs_obs:
            for s in traj:
                freq[s] += 1

        sum = 0
        for traj in self.trajs_obs:
            sum += 1

        svf_1 = list(freq / sum)

        expert_fe_1 = np.array(self.fm_1.T.dot(svf_1))

        return expert_fe_1





    def get_expert_2(self):
        """Determine the second expert feature expectation based on observed traffic volume data
           The approximated population flow was obtained using the scaled-up trajectories

        :return: expert_fe_2: the second expert feature expectation
        """

        freq = np.zeros(self.nS)

        for i in range(self.nS):
            if(str(i) in list(self.counts.keys())):
                freq[i] += self.counts[str(i)]
            else:
                freq[i] = 0

        sum = 0
        for traj in self.trajs_scaled:
            sum += 1

        svf_2 = list(freq / sum)

        expert_fe_2 = np.array(self.fm_2.T.dot(svf_2))

        return expert_fe_2
