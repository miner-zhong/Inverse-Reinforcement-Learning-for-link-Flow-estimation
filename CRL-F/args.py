


def appropo_args(parser):
    """Set parameters for CRL-F
    """

    parser.add_argument("--num_epochs", type=int, default=100)

    # computing devices either 'cpu' or 'cuda'
    parser.add_argument('--device', type=str, default='cpu')


    # number of policies in the cache
    parser.add_argument("--cache_size", type=int, default=5)


    # parameters of the projection oracle
    parser.add_argument("--proj_lr", type=float, default=1)     # learning rate used in the projection oracle
    parser.add_argument('--mx_size', type=int, default=20)      # value for transforming a convex set into a convex cone

    # parameters of the projection oracle
    parser.add_argument("--rl_traj", type=int, default=60)                 # number of trajectories generated each time entering this oracle
    parser.add_argument("--rl_iter", type=int, default=10)                 # number of maximum steps in each generated trajectory
    parser.add_argument("--rl_lr", type=float, default=1e-4)               # RL learning rate
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G')  # RL discount factor
    parser.add_argument("--value_coef", type=float, default=1.0)            # weight of value in RL objection
    parser.add_argument("--entropy_coef", type=float, default=0.001)        # weight of entropy in RL objective
    parser.add_argument("--entropy_patience", type=float, default=2)        # patience for checking entropy changes
    parser.add_argument("--entropy_threshold", type=float, default=0.0001)  # threshold for checking entropy changes
    parser.add_argument("--entropy_increase", type=float, default=0.001)    # increase in weight of entropy


    # thresholds used in the constraints
    parser.add_argument("--obs_counts_threshold", type=float, default=0.0001)
    parser.add_argument("--obs_trajs_threshold", type=float, default=0.0001)


    # stopping criteria
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--thres', type=int, default=1e-6)


    # input & output data
    parser.add_argument('--network_property_file', type=str, default='./INPUT/network_properties.csv')
    parser.add_argument('--trajectory_file', type=str, default= './INPUT/trajectory_file.csv')
    parser.add_argument('--transition_info_file', type=str, default= './INPUT/transition_information.csv')
    parser.add_argument('--state_feature_file', type=str, default= './INPUT/state_feature_file.csv')
    parser.add_argument('--traffic_count_file', type=str, default= './INPUT/traffic_count_file.csv')
    parser.add_argument('--output_svf', type=str, default='./RESULT/svf_record.csv')
    parser.add_argument('--output_trajs', type=str, default='./RESULT/trajs_record.csv')

