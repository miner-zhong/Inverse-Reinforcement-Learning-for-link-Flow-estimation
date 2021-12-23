from algo_rl.olo import OGD
import numpy as np


class ProjectionOracle:
    """Projection oracle (update theta)

    Arguments:
        dim: dimension of theta
        olo: oline gradient descent algorithm
        args: parameters
        env: road network
    """


    def __init__(self, dim=None, args=None, env=None):
        self.dim = dim
        self.olo = OGD(dim+1, self.proj_decision_set, args.proj_lr)
        self.args= args
        self.env = env



    def proj(self, p, args, env):
        """Project current average measurement vector to the constraint set
        """
        p = np.ndarray.copy(p)

        # constraint: feature expectation matching -- trajectories
        diff_1 = p[:env.fm_1.shape[1]] - env.expert_fe_1
        if np.linalg.norm(diff_1) > args.obs_trajs_threshold:
            p[:env.fm_1.shape[1]] = (diff_1 / np.linalg.norm(diff_1)) * args.obs_trajs_threshold + env.expert_fe_1

        # constraint: feature expectation matching -- traffic count
        diff_2 = p[env.fm_1.shape[1]:] - env.expert_fe_2
        if np.linalg.norm(diff_2) > args.obs_counts_threshold:
            p[env.fm_1.shape[1]:] = (diff_2 / np.linalg.norm(diff_2)) * args.obs_counts_threshold + env.expert_fe_2

        return p



    def g(self, p=None, alpha=None):
        p_on_plane = p[:-1]
        q = self.proj((self.args.mx_size / alpha)*p_on_plane, self.args, self.env)
        q *= (alpha/self.args.mx_size)
        q = np.append(q, alpha)
        return np.linalg.norm(q-p), q


    def proj_cone(self, p):
        epsilon = 1e-6
        left = 0.0
        right = 100
        while np.abs(right-left) > epsilon:
            m1 = left + (right - left)/3
            m2 = right - (right - left)/3
            if self.g(p, m1)[0] < self.g(p, m2)[0]:
                right = m2
            else:
                left = m1
        return self.g(p, (left+right)/2.0)[1]



    def proj_polar_cone(self, p):
        q = self.proj_cone(p)
        return p-q


    def proj_decision_set(self, p):
        for i in range(self.args.mx_size):
            p = self.proj_polar_cone(p)
            if np.linalg.norm(p) > 1:
                p = p / np.linalg.norm(p)
        return p


    def update(self, expected_return):
        expected_return = np.append(expected_return, self.args.mx_size)
        loss_vector = -expected_return
        self.olo.step_linear(loss_vector)


    def reset(self):
        self.olo.reset()


    def get_theta(self):
        return self.olo.get_theta()





