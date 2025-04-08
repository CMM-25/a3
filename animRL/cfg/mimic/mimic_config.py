# created by Fatemeh Zargarbashi - 2025

from animRL.cfg.base.base_config import BaseEnvCfg, BaseTrainCfg


class MimicCfg(BaseEnvCfg):
    class env(BaseEnvCfg.env):
        num_observations = 1  # should be overwritten
        num_actions = 1  # should be overwritten
        num_privileged_obs = None  # None

        episode_length = 250  # episode length

        reference_state_initialization = False  # initialize state from reference data

        play = False
        debug = False

    class motion_loader:
        motion_files = ''

    class init_state(BaseEnvCfg.init_state):
        pos = [0.0, 0.0, 0.9]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        added_height = 0.02  # height added to root when rsi

        # default joint angles =  target angles [rad] when action = 0.0
        default_joint_angles = {
            "joint_lowerback_x": 0.0,
            "joint_lowerback_z": 0.0,
            "joint_upperback_x": 0.0,
            "joint_upperback_y": 0.0,
            "joint_upperback_z": 0.0,
            "joint_lowerneck_y": 0.0,
            "joint_lScapula_y": 0.0,
            "joint_lShoulder_1": 0.0,
            "joint_lShoulder_2": 0.13,
            "joint_lShoulder_torsion": -0.5,
            "joint_lElbow_flexion_extension": -0.25,
            "joint_lElbow_torsion": 0.0,
            "joint_lWrist_x": 0.0,
            "joint_lWrist_z": -0.2,
            "joint_rScapula_y": 0.0,
            "joint_rShoulder_1": 0.0,
            "joint_rShoulder_2": -0.13,
            "joint_rShoulder_torsion": 0.5,
            "joint_rElbow_flexion_extension": -0.25,
            "joint_rElbow_torsion": 0.0,
            "joint_rWrist_x": 0.0,
            "joint_rWrist_z": 0.2,
            "joint_lHip_1": 0.0,
            "joint_lHip_2": 0.0,
            "joint_lHip_torsion": 0.0,
            "joint_lKnee": 0.0,
            "joint_lAnkle_1": 0.0,
            "joint_rHip_1": 0.0,
            "joint_rHip_2": 0.0,
            "joint_rHip_torsion": 0.0,
            "joint_rKnee": 0.0,
            "joint_rAnkle_1": 0.0
        }

    class control(BaseEnvCfg.control):
        control_type = 'P'  # P: position, V: velocity, T: torques
        stiffness = {'joint': 300.}  # [N*m/rad]
        damping = {'joint': 50.0}  # [N*m*s/rad]

        # action scale: target angle = actionScale * action + defaultAngle
        scale_joint_target = 0.25
        clip_joint_target = 100.
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(BaseEnvCfg.asset):
        file = '{ROOT_DIR}/resources/robots/bob_v3/bob_v3.urdf'
        terminate_after_contacts_on = ["pelvis", "UpperLeg", "lumbar", "torso", "Scapula", "UpperArm", "neck", "head"]

        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        ee_offsets = {
            "lFoot": [0.07, 0.0, -0.02],
            "rFoot": [0.07, 0.0, -0.02],
        }
        collapse_fixed_joints = True

    class rewards(BaseEnvCfg.rewards):
        class terms:
            # sigma, tolerance
            joint_targets_rate = [10.0, 0.0]

        soft_dof_pos_limit = 0.9  # percentage of urdf limits, values above this limit are penalized

    class domain_rand(BaseEnvCfg.domain_rand):
        randomize_friction = False
        friction_range = [0.25, 1.75]

        randomize_base_mass = False
        added_mass_range = [-1., 1.]

    class termination:
        max_base_lin_vel = 10.0
        max_base_ang_vel = 100.0
        max_height = 3.0

    class viewer(BaseEnvCfg.viewer):
        enable_viewer = False
        overview = True
        record_camera_imgs = False

        # Note: if the viewer is disabled, the following parameters are ignored, only the quadruped will be recorded in
        # the video. If the viewer is enabled, the following parameters are used to configure the viewer.
        vis_flag = ['ref_only']
        ref_pos_b = [2, 2, 2.65]
        camera_pos_b = [2., 2., 2.5]

class MimicTrainCfg(BaseTrainCfg):
    algorithm_name = 'PPO'

    class policy:
        log_std_init = 0.0
        actor_hidden_dims = [512, 256]
        critic_hidden_dims = [512, 256]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        surrogate_coef = 1
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs * num_steps / num_minibatches
        learning_rate = 1.e-3
        schedule = 'fixed'  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner:
        num_steps_per_env = 24  # per iteration
        max_iterations = 5000  # number of policy updates
        normalize_observation = True
        save_interval = 100  # check for potential saves every this many iterations

        record_gif = True  # need to enable env.viewer.record_camera_imgs and run with wandb
        record_gif_interval = 100
        record_iters = 10  # should be int (* num_steps_per_env)

        # logging
        run_name = 'test' # name of each run
        experiment_name = 'bob'

        # load and resume
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model

        wandb = True
        wandb_group = "default"
