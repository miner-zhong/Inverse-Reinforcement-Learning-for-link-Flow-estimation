import numpy as np

from gym.utils import seeding


def categorical_sample(prob_n, np_random):
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()


class DiscreteEnv():
    """A simple environment with MDP properties

    Attributes:
        nS: number of states
        nA: number of actions
        P: transition probabilities
        isd: initial state distribution
        s: a chosen initial state
    """

    def __init__(self, nS, nA, P, isd):
        self.nS = nS
        self.nA = nA
        self.P = P
        self.isd = isd

        self.seed()
        self.s = categorical_sample(self.isd, self.np_random)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """Reset the MDP to a starting state
        """
        self.s = categorical_sample(self.isd, self.np_random)
        return self.s

    def step(self, a):
        """Choose the next state given current state and action
        """
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d= transitions[i]
        self.s = s
        return (s, r, d, {"prob" : p})



