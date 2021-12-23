import copy
import torch
from collections import namedtuple


CacheItem = namedtuple('CacheItem', ['exp_rtn', 'policy', 'exp_trajs'])


def init_cache(rl_oracle_generator=None, args=None):
    """Initialize a cache of policies, used as warm-start for the CRL-F algorithm

    :param rl_oracle_generator: RL oracle generator
    :param args: parameters
    :return: cache: a cache of initial policies
    """

    cache = []

    for _ in range(args.cache_size):
        rl_oracle = rl_oracle_generator()

        with torch.no_grad():
            [exp_rtn, ls_trajs] = rl_oracle.learn_policy(n_traj=args.rl_traj, n_iter=args.rl_iter,
                                                          update=False, ls_measurements=[], args = args)

        cache.append(CacheItem(copy.deepcopy(exp_rtn), rl_oracle.net.state_dict(), copy.deepcopy(ls_trajs)))

    return cache


