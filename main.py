import algo_irl
import roadnetwork

import csv

import argparse
from args import irl_args
parser = argparse.ArgumentParser()
irl_args(parser)
args = parser.parse_args()




def main(epochs, learning_rate):
    """initialize environment and oracles, the run IRL-F.

    Output: estimated state visitation frequencies based on the optimal policy determined

    :param epochs: number of iterations in IRL-F
    :param learning_rate: learning rate in IRL-F
    """

    # initialize a RoadNetwork using input data

    rn = roadnetwork.RoadNetwork(args)


    # open a file for output state visitation frequncies
    # write the ground-truth state visitation frequencies in the first row

    with open(args.output_svf_file, 'w', newline='') as f:
        head = ['iteration']
        for i in range(rn.nb_state):
            head.append('s' + str(i))
        writer = csv.writer(f)
        writer.writerow(head)
        writer.writerow(['truth'] + rn.truth_svf)

    # run IRL-F

    algo_irl.irl(rn, args, epochs, learning_rate)

    print('done')





if __name__ == '__main__':
    main(args.iteration, args.learning_rate)
