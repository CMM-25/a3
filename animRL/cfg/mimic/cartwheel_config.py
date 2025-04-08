from animRL.cfg.mimic.mimic_config import MimicCfg, MimicTrainCfg


class CartwheelCfg(MimicCfg):
    class env(MimicCfg.env):
        num_envs = 4096

        num_actions = 32
        num_observations = 108

        episode_length = 100  # episode length

        reference_state_initialization = True  # initialize state from reference data

    class motion_loader(MimicCfg.motion_loader):
        motion_files = '{ROOT_DIR}/resources/datasets/bob3/Cartwheel.txt'

    class rewards(MimicCfg.rewards):
        class terms:

            # ----------- TODO 2.3: tune the hyperparameters
            # reward_name = [sigma, tolerance]
            joint_targets_rate = [1.0, 0.0]

            track_base_height = [1.0, 0.0]
            track_base_orientation = [1.0, 0.0]
            track_joint_pos = [1.0, 0.0]
            track_base_vel = [1.0, 0.0]
            track_ee_pos = [1.0, 0.0]
            # ----------- End of implementation

    class control(MimicCfg.control):
        control_type = 'P'  # P: position, V: velocity, T: torques
        stiffness = {'joint': 300.}  # [N*m/rad]
        damping = {'joint': 50.0}  # [N*m*s/rad]


class CartwheelTrainCfg(MimicTrainCfg):
    algorithm_name = 'PPO'

    class runner(MimicTrainCfg.runner):
        run_name = 'cartwheel'
        experiment_name = 'bob'
        max_iterations = 5000  # number of policy updates

    class algorithm(MimicTrainCfg.algorithm):

        # ----------- TODO 2.3: tune the hyperparameters
        learning_rate = 1.e-3
        schedule = 'fixed'

        entropy_coef = 0.01
        value_loss_coef = 0.5
        clip_param = 0.2
        desired_kl = 0.01

        bootstrap = True
        # ----------- End of implementation

    class policy(MimicTrainCfg.policy):

        # ----------- TODO 2.3: tune the hyperparameters
        log_std_init = 0
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # ----------- End of implementation