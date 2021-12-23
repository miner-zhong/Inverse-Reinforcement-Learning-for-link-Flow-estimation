import numpy as np
from numpy import linalg as LA
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional
import torch.nn.functional as F
from torch.distributions import Categorical



SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class RL_Oracle:
    """Reinforcement Learning oracle (find policy)

    given the rewards in MDP, find a single policy and compute the average measurement vector

    Arguments:
        env: road network
        gamma: discount factor
        lr: learning rate
        saved_actions: a tuple used to save actions made during the excecution
        rewards: rewards achieved during the excedution
        entropies: entropies achieved during the excedution
        theta: a variable in CRL-F, it is fixed each time running the RL oracle
        device: where the neural network is trained
        net: the neural network (as policy)
        optimizer: neural network optimizer
        entropy_coef: coefficient of the entropy in the RL objective
        value_coef: coefficient of the value in the RL objective
        args: parameters
    """


    def __init__(self, env=None, theta=None, net=None, args=None):
        super(RL_Oracle, self).__init__()
        self.env = env
        self.gamma = args.gamma
        self.lr = args.rl_lr
        self.saved_actions = []
        self.rewards = []
        self.entropies = []
        self.theta = theta
        self.device = args.device
        self.net = net
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.entropy_coef = args.entropy_coef
        self.value_coef = args.value_coef
        self.args=args



    def reset(self):
        del self.saved_actions[:]
        del self.rewards[:]
        del self.entropies[:]



    def select_action(self, state, forbid_action):
        """Select an action for a given state based on current policy

        :param state: the current state
        :param forbid_action: actions that need to be masked
        :return: a selected action
        """

        state = torch.from_numpy(state).float().to(self.device)
        action_scores, state_value = self.net(state)

        if(len(forbid_action)>0):
            for a in forbid_action:
                action_scores[:, a] = 1e+10
        m = Categorical(logits=-action_scores)
        action = m.sample()

        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))
        self.entropies.append(m.entropy())
        return action.item()




    def finish_episode(self):
        """update the policy given the generated trajectories and rewards achieved
        """
        R = 0
        saved_actions = self.saved_actions
        policy_losses = []
        value_losses = []
        returns = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)

        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()
            policy_losses.append(-log_prob * advantage)
            R = torch.tensor([R]).to(self.device)
            value_losses.append(F.smooth_l1_loss(value, R.reshape(-1,1)))

        self.optimizer.zero_grad()

        loss = torch.stack(policy_losses).mean()\
             + (self.value_coef * torch.stack(value_losses).mean())\
             - (self.entropy_coef * torch.stack(self.entropies).mean())

        loss.backward()

        nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)

        self.optimizer.step()

        self.reset()





    def learn_policy(self, n_traj, n_iter, update=True, ls_measurements = None, args = None):
        """Find a single policy given current rewards,
           Generate trajectories based on this policy and compute the average measurement vector

        :param n_traj: number of trajectories generated
        :param n_iter: maximum number of steps in each generated trajectory
        :param update: if True, update the policy using generated trajectories, otherwise, skip this step
        :param ls_measurements: a list of average measurement vectors from previous policies
        :param args: parameters
        :return:
        """

        self.reset()

        sum_measurements = np.zeros(np.shape(self.theta))

        ls_trajs = []

        # generate trajectories under current policy
        for _ in range(n_traj):
            current_state = self.env.reset()
            ls_vec = []
            traj = []

            for step in range(n_iter):
                current_state_location = self.env.state_from_repr_to_idx(current_state)

                invalid_action = []
                for a in range(self.env.nA):
                    if (self.env.P[current_state_location][a][0][1] == current_state_location):
                        invalid_action.append(a)

                action = self.select_action(current_state, forbid_action=invalid_action)

                current_state, done = self.env.step(action)

                next_state_location = self.env.state_from_repr_to_idx(current_state)

                # measurement vector for this step in this trajectory
                measurements = np.zeros(self.env.fm_1.shape[1] + self.env.fm_2.shape[1])
                measurements[:self.env.fm_1.shape[1]] = self.env.fm_1[current_state_location]
                measurements[self.env.fm_1.shape[1]:] = self.env.fm_2[current_state_location]

                # stacked measurement vectors used for calculating reward
                ls_vec.append(measurements)

                traj.append(current_state_location)
                if done:
                    traj.append(next_state_location)
                    ls_trajs.append(traj)
                    break

            for m in range(len(ls_vec)):
                current_measure = ls_vec[m]
                current_theta = self.theta
                reward = np.dot(current_theta, current_measure)
                reward = -reward
                self.rewards.append(reward)

            # update policy (used during training, but deactivate during cache initialization)
            if update:
                self.finish_episode()

            traj_measurements = np.zeros(self.theta.size)
            for vec in ls_vec:
                traj_measurements += vec

            sum_measurements = sum_measurements + traj_measurements

        avg_measurements = sum_measurements / n_traj



        # if average measurement vector of this policy is simialr to several other previous policies, increase exploration in RL

        if (len(ls_measurements) > args.entropy_patience):
            ls_check = []
            for i in range(args.entropy_patience):
                ls_check.append(LA.norm(ls_measurements[-1 * (i + 1)] - avg_measurements, ord=2) < args.entropy_threshold)
            check = True
            for i in range(args.entropy_patience):
                if (ls_check[i] == False):
                    check = False
            if (check):
                self.entropy_coef = self.entropy_coef + args.entropy_increase
            else:
                self.entropy_coef = args.entropy_coef


        return avg_measurements, ls_trajs


