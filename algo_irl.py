from itertools import product
import numpy as np
import csv



def irl(rn, args, epochs, learning_rate):
    """IRL-F algorithm

    :param rn: road network initialized with input data files
    :param args: parameters for IRL-F
    :param epochs: number of iterations
    :param learning_rate: learning rate
    """

    # initialize alpha (reward weights)
    alpha = np.random.uniform(size=(rn.feature_matrix.shape[1],))


    # calculate expert feature expectation
    true_exp = find_feature_expectations(rn.feature_matrix, rn.trajs_obs, rn.trajs_scaled, rn.counts, rn.nb_state)


    # to record all state visitation frequencies generated in each iteration
    all_svf = {}


    for i in range(epochs):

        # calculate reward based on current alpha and feature matrix
        r = rn.feature_matrix.dot(alpha)


        # calculate policy state visitation frequencies
        policy_svf = find_expected_svf(rn.n_states, r, rn.n_actions, args.discount,
                                         rn.transition_probability, rn.trajs_obs, args.trajectory_horizon)


        # calculate policy feature expectation
        policy_exp = calculate_expected_feature_expectation(rn.feature_matrix, rn.feature_matrix_1, rn.feature_matrix_2, policy_svf)


        # update alpha
        grad = true_exp - policy_exp
        alpha += learning_rate * grad


        # write current state visitation frequencies to output file
        with open(args.output_svf_file, 'a', newline='') as f:
            current_svf = []
            for j in range(rn.nb_state):
                current_svf.append(policy_svf[j])

            current_svf_normalized = [str(i)]
            for j in range(rn.nb_state):
                current_svf_normalized.append(current_svf[j] / sum(current_svf))

            all_svf[str(i)] = current_svf_normalized

            writer = csv.writer(f)
            writer.writerow(current_svf_normalized)


        # stop if the change in state visitation frequencies is smaller than the pre-determined threshold
        if(i > args.patience):
            patience = args.patience

            nSVF = all_svf[str(i)][1:]
            oSVF = all_svf[str(i-patience)][1:]

            check = 0
            for a in range(len(nSVF)):
                if(nSVF[a] - oSVF[a] > check):
                    check = nSVF[a] - oSVF[a]

            if(check < args.thres):
                print('REACH STOP CRITERIA - SMALL IMPROVMENT')
                break

    print('REACH ITERATION LIMIT')








def find_feature_expectations(feature_matrix, obs_trajectories, scaled_trajectories, traffic_counts, nb_state):
    """Calculate expert feature expectation

    :param feature_matrix: complete feature matrix
    :param obs_trajectories: array of observed trajectories
    :param scaled_trajectories: array of trajectories scalled-up to determined approximated population flow
    :param traffic_counts: observed traffic volumes
    :param nb_state: number of links
    :return: expert feature expectation
    """

    feature_expectations = np.zeros(feature_matrix.shape[1])

    for trajectory in obs_trajectories:
        for state in trajectory:
            feature_expectations[:nb_state] += feature_matrix[state][:nb_state]
    feature_expectations[:nb_state] /= obs_trajectories.shape[0]

    for k in traffic_counts.keys():
        feature_expectations[nb_state:] += (feature_matrix[int(k)][nb_state:] * traffic_counts[k])
    feature_expectations[nb_state:] /= scaled_trajectories.shape[0]

    return feature_expectations






def calculate_expected_feature_expectation(feature_matrix, feature_matrix_1, feature_matrix_2, expected_svf):
    """Calculate policy feature expectation

    :param feature_matrix: complete feature matrix
    :param feature_matrix_1: partial feature matrix (number of columns = dimension of the first state feature vector)
    :param feature_matrix_2: partial feature matrix (number of columns = dimension of the second state feature vector)
    :param expected_svf: policy state visitation frequencies
    :return: policy feature expectation
    """

    expected_feature = np.zeros(feature_matrix.shape[1])

    expected_feature[:feature_matrix_1.shape[1]] = feature_matrix_1.T.dot(expected_svf)

    expected_feature[feature_matrix_1.shape[1]:] = feature_matrix_2.T.dot(expected_svf)

    return expected_feature





def find_expected_svf(n_states, r, n_actions, discount, transition_probability, trajectories, trajectory_horizon):
    """Calculate policy state visitation frequencies

    :param n_states: number of all states (number of links + number of aborbing states)
    :param r: current rewards
    :param n_actions: number of all actions
    :param discount: discount factor
    :param transition_probability
    :param trajectories: array of observed trajectories
    :param trajectory_horizon: maximum number of steps in generated trajectories
    :return: policy state visitation frequencies
    """

    n_trajectories = trajectories.shape[0]
    trajectory_length = trajectory_horizon

    policy = find_policy(n_states, n_actions, transition_probability, r, discount)

    start_state_count = np.zeros(n_states)
    for trajectory in trajectories:
        start_state_count[trajectory[0]] += 1
    p_start_state = start_state_count/n_trajectories


    expected_svf = np.tile(p_start_state, (trajectory_length, 1)).T
    for t in range(1, trajectory_length):
        expected_svf[:, t] = 0
        for i, j, k in product(range(n_states), range(n_actions), range(n_states)):
            expected_svf[k, t] += (expected_svf[i, t-1] *
                                  policy[i, j] *
                                  transition_probability[i, j, k])

    return expected_svf.sum(axis=1)






def find_policy(n_states, n_actions, transition_probabilities, reward, discount):
    """Determine policy based on current rewards

    Following equation 9.2 from Ziebart's thesis.]

    :param n_states: number of all states
    :param n_actions: number of all actions
    :param transition_probabilities
    :param reward
    :param discount: discount factor
    :return: Q value
    """

    value = np.zeros(n_states)
    diff = float("inf")
    while diff > 1e-2:
        diff = 0
        for s in range(n_states):
            max_v = float("-inf")
            for a in range(n_actions):
                tp = transition_probabilities[s, a, :]
                max_v = max(max_v, np.dot(tp, reward + discount*value))

            new_diff = abs(value[s] - max_v)
            if new_diff > diff:
                diff = new_diff
            value[s] = max_v

    Q = np.zeros((n_states, n_actions))
    for i in range(n_states):
        for j in range(n_actions):
            p = transition_probabilities[i, j, :]
            Q[i, j] = p.dot(reward + discount*value)
    Q -= Q.max(axis=1).reshape((n_states, 1))
    Q = np.exp(Q)/np.exp(Q).sum(axis=1).reshape((n_states, 1))
    return Q



