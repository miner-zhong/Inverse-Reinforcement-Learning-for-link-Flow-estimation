
import csv
import numpy as np

from RoadNetwork import RoadEnv
from algo_rl.projection_oracle import ProjectionOracle
from algo_rl.rl_oracle import RL_Oracle
from algo_rl.nets import MLP
import algo_rl.solver as solver

import argparse
from args import appropo_args
parser = argparse.ArgumentParser()
appropo_args(parser)
args = parser.parse_args()




def main():
    """ initialize environment and oracles, the run CRL-F.

    Output: estimated state visitation frequencies based on the mixture policy that satisfy the constraints
    """

    # initialize the road network with input data
    env = RoadEnv(args)
    env.reset()


    # initialize neural network (used to determine the policy in MDP)
    net = MLP(env)
    net = net.to(args.device)


    # initialize theta (lambda in the original paper)
    theta = np.zeros(env.fm_1.shape[1] + env.fm_2.shape[1])


    # initialize the reinforcement learning oracle
    rl_oracle_generator = lambda: RL_Oracle(env=env, net=net, args=args, theta=theta)


    # initialize the projection oracle
    proj_oracle = ProjectionOracle(dim=theta.size, args=args, env=env)


    # open a file for output state visitation frequencies
    # write the ground-truth state visitation frequencies in the first row

    with open(args.output_svf, 'w', newline='') as f:
        head = ['iteration']
        for j in range(env.nb_state):
            head.append('s' + str(j))
        firstline = ['truth'] + env.truth_svf
        writer = csv.writer(f)
        writer.writerow(head)
        writer.writerow(firstline)


    # run CRL-F
    solver.run(proj_oracle=proj_oracle, rl_oracle_generator=rl_oracle_generator, args=args, env=env)


    print('done')



if __name__ == "__main__":
    main()

