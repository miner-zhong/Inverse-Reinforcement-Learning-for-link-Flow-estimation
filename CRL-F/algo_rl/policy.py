import numpy as np


class MixturePolicy:
    """mixed policy where the average measurement vectors and generated trajectories of each single policy are recorded

    Arguments:
        loss_vec: a list of average measurement vector achieved during each single policy
        ls_trajectories: a list of trajectories generated during each single policy
        env: road network
    """

    def __init__(self, env=None):
        self.loss_vec = []
        self.ls_trajectories = []
        self.env = env



    def add_response(self, best_exp_rtn=None, trajs=None):
        """Add a new single policy to the mixture policy

        :param best_exp_rtn: average measurement vector of this signle policy
        :param trajs: trajectories generated during this single policy
        """

        self.loss_vec.append(best_exp_rtn)

        self.exp_rtn_of_avg_policy = np.average(np.stack(self.loss_vec, axis=0), axis=0)

        self.ls_trajectories.append({'traj_list': trajs, 'avg_measurement': best_exp_rtn})



