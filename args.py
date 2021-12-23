

def irl_args(parser):
    """Set parameters for IRL-F
    """
    parser.add_argument('--iteration', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--trajectory_horizon', type=int, default=10)

    # stopping criteria
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--thres', type=int, default=1e-6)

    # input data file
    parser.add_argument('--network_property_file', type=str,
                        default= './INPUT/network_properties_file.csv')

    parser.add_argument('--state_feature_file', type=str,
                        default= './INPUT/state_feature_file.csv')

    parser.add_argument('--trajectory_file', type=str,
                        default= './INPUT/trajectory_file.csv')

    parser.add_argument('--traffic_count_file', type=str,
                        default= './INPUT/traffic_count_file.csv')

    parser.add_argument('--transition_info_file', type=str,
                        default= './INPUT/transition_information.csv')

    # output data file
    parser.add_argument('--output_svf_file', type=str, default='./RESULT/svf_record.csv')


