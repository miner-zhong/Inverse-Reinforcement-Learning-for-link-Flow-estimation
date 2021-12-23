import numpy as np
import csv

from algo_rl.cache import init_cache, CacheItem
from algo_rl.policy import MixturePolicy




def run(proj_oracle=None, rl_oracle_generator=None, args=None, env=None):
    """CRL-F algorithm

    :param proj_oracle: initialized projection oracle
    :param rl_oracle_generator: initialized projection oracle generator
    :param args: parameters
    :param env: initialized road environment
    """


    # initialize a mixture policy
    policy = MixturePolicy(env)


    # initialize a cache with several initial policies
    cache = init_cache(rl_oracle_generator=rl_oracle_generator, args=args)
    init = True
    new_cache_item = True
    value = float("inf")


    # get current theta
    theta = proj_oracle.get_theta()


    # to check state visitation frequencies updates
    ls_svf = []


    # to control exploration in RL oracle
    ls_avg_measurement = []


    for episode in range(args.num_epochs):

        min_value = float("inf")
        min_exp_rtn = None
        min_params = None
        min_exp_trajs = []
        reset = False

        # if initialization or the cache was updated in last iteration(value<0), find the min-value cached policy
        if value < 0 or np.isclose(0, value, atol=0.0, rtol=1e-05) or init:
            reset= True
            for item in cache:
                value = np.dot(theta, np.append(item.exp_rtn, args.mx_size))
                if value < min_value or init:
                    min_value = value
                    min_exp_rtn = item.exp_rtn
                    min_params = item.policy
                    min_exp_trajs = item.exp_trajs
                    init=False


        # if initialization or the cache was updated, use the min-value cached policy to warm start
        # else, Run RL Oracle to find a new policy
        if reset:
            best_exp_rtn = min_exp_rtn
            exp_trajs = min_exp_trajs
            rl_oracle = rl_oracle_generator()
            new_params = rl_oracle.net.state_dict()
            new_params.update(min_params)
            rl_oracle.net.load_state_dict(new_params)
            rl_oracle.theta = theta[:-1]
        else:
            rl_oracle.theta = theta[:-1]
            [best_exp_rtn, exp_trajs] = rl_oracle.learn_policy(n_traj=args.rl_traj, n_iter=args.rl_iter,
                                                               update = True, ls_measurements=ls_avg_measurement, args=args)
            ls_avg_measurement.append(best_exp_rtn)
            new_cache_item = True



        # if current policy have value<0, meaning this 'positive response oracle' is finished (the policy is good enough),
        # then move on to projection oracle, update theta, add policy to cache, update policy info
        # oherwise, the policy is not good enough, no update, go to next episode, find a new policy

        value = np.dot(theta, np.append(best_exp_rtn, args.mx_size))

        if value < 0 or np.isclose(0, value, atol=0.0, rtol=1e-05):

            # update theta in the projection oracle
            proj_oracle.update(best_exp_rtn.copy())
            theta = proj_oracle.get_theta()

            # add the new single policy to mixture policy
            policy.add_response(best_exp_rtn=best_exp_rtn, trajs=exp_trajs)

            # add the new single policy to cache
            if new_cache_item:
                cache.append(CacheItem(best_exp_rtn, rl_oracle.net.state_dict(), exp_trajs))
                new_cache_item = False

            # check current distance to the constraint set
            dist_to_target = np.linalg.norm(policy.exp_rtn_of_avg_policy - proj_oracle.proj(policy.exp_rtn_of_avg_policy, args=args, env=env))

            status = "UPDATE THETA, current dist-to-target: {}\n".format(dist_to_target)


            # calculate current state visitation frequency and wrtite it to the ouput file
            svf = np.zeros(env.nb_state)
            sum = 0
            for info in policy.ls_trajectories:
                for traj in info['traj_list']:
                    for s in traj:
                        if (s < env.nb_state):
                            repr = np.zeros(env.nb_state)
                            repr[int(s)] = 1
                            svf += repr
                    sum += 1
            svf /= sum
            current_svf = list(svf)

            with open(args.output_svf, 'a', newline='') as f:
                ls_svf.append(current_svf)
                current = [str(episode)] + current_svf
                writer = csv.writer(f)
                writer.writerow(current)


            # if no significant svf updates in previous few episodes, finish
            if (len(ls_svf) > args.patience):
                nsvf = ls_svf[-1]
                osvf = ls_svf[-1-args.patience]
                check = 0
                for k in range(len(nsvf)):
                    if (nsvf[k] - osvf[k] > value):
                        check = nsvf[k] - osvf[k]
                if (check < args.thres):
                    print('REACH STOP CRITERIA - SMALL IMPROVMENT')
                    break

            # if current mixed-policy have average measurement have a very small distance to the constraint set, finish
            if dist_to_target < 0 or np.isclose(0, dist_to_target, atol=0.0, rtol=1e-05):
                print('OPTIMAL RESULTS FOUND')
                break

        # otherwise, return to RL oracle to find a new policy
        else:
            status = 'NO UPDATE, policy not good enough' + '\n'

        print('episode-' + str(episode) + '\n' + status + '\n')


    print('REACH ITERATION LIMIT')












