# created by Fatemeh Zargarbashi - 2025
import isaacgym
from matplotlib.animation import FuncAnimation, FFMpegWriter

from animRL import ROOT_DIR
from animRL.utils.helpers import get_load_path, update_cfgs_from_dict
from animRL.utils.isaac_helpers import get_args
from animRL.envs import task_registry
from animRL.utils.plots import forward_kinematics, plot_robot

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from urdfpy import URDF

"""
This code is used to evaluate the trained policy on the character. Run it with your corresponding task name and load_run. 
It will save a plot of the rewards, a video of the motion, and a json file containing the observations, rewards, and dones.
"""


class Eval:
    """ Evaluation script to play the policy."""

    def __init__(self, args, seed=1):
        self.args = args

        env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

        if args.load_run is not None:
            train_cfg.runner.load_run = args.load_run
        if args.checkpoint is not None:
            train_cfg.runner.checkpoint = args.checkpoint
        load_path = get_load_path(
            os.path.join(ROOT_DIR, "logs", train_cfg.runner.experiment_name),
            load_run=train_cfg.runner.load_run,
            checkpoint=train_cfg.runner.checkpoint,
        )
        self.load_path = load_path
        print(f"Loading model from: {load_path}")

        # load config
        load_config_path = os.path.join(os.path.dirname(load_path), f"{train_cfg.runner.experiment_name}.json")
        with open(load_config_path) as f:
            load_config = json.load(f)
            update_cfgs_from_dict(env_cfg, train_cfg, load_config)

        # overwrite config params
        env_cfg.seed = seed
        env_cfg.env.num_envs = 1

        env_cfg.env.play = True
        env_cfg.sim.use_gpu_pipeline = False
        env_cfg.env.episode_length = 1000  # to prevent timeout
        env_cfg.viewer.overview = False

        self.env_cfg = env_cfg
        self.train_cfg = train_cfg

        # prepare environment, runner and policy
        env, _ = task_registry.make_env(name=self.args.task, args=self.args, env_cfg=self.env_cfg)
        self.env = env
        self.runner = task_registry.make_alg_runner(env=env, name=self.args.task, args=self.args, env_cfg=self.env_cfg,
                                                    train_cfg=self.train_cfg)
        self.env.reset()
        self.obs = self.env.get_observations()
        self.runner.load(self.load_path)  # load policy
        self.policy = self.runner.get_inference_policy(device=self.env.device)
        self.info = None
        self.buf = {'obs': [],
                    'dones': [],
                    'rewards': {}
                    }
        self.plot_buf = {
            'joint_angles': [],
            'base_pos': [],
            'base_quat': [],
            'ref_joint_angles': [],
            'ref_base_pos': [],
            'ref_base_quat': [],
        }

    def play(self):
        self.env.is_playing = True
        x_getter = self.env.get_time_stamp
        y_getters = (self.env.getplt_rewards,)
        self.env.plotter_init(y_getters)

        # rollout
        env_id = 0
        for i in range(200):
            actions = self.policy(self.obs.detach())

            self.obs, _, _, self.dones, self.info = self.env.step(actions.detach())
            # if self.dones[env_id]:
            #     break

            self.buf['obs'].append(self.obs[env_id].numpy().tolist())
            self.buf['dones'].append(self.dones[env_id].numpy().tolist())
            for key in self.env.rew_terms_buf:
                if key not in self.buf['rewards']:
                    self.buf['rewards'][key] = []
                self.buf['rewards'][key].append(self.env.rew_terms_buf[key][env_id].numpy().tolist())

            self.plot_buf['joint_angles'].append(
                {'names': self.env.dof_names, 'values': self.obs[env_id, 11:11 + 32].clone().numpy()})
            self.plot_buf['base_pos'].append(self.env.root_states[env_id, :3].clone().numpy())
            self.plot_buf['base_quat'].append(self.env.root_states[env_id, 3:7].clone().numpy())

            target_frame = self.env.data['target_frames'][env_id]
            self.plot_buf['ref_joint_angles'].append(
                {'names': self.env.dof_names,
                 'values': self.env.motion_loader.get_joint_pose(target_frame).clone().numpy()})
            self.plot_buf['ref_base_pos'].append(self.env.motion_loader.get_root_pos(target_frame).clone().numpy())
            self.plot_buf['ref_base_quat'].append(self.env.motion_loader.get_root_rot(target_frame).clone().numpy())

            self.env.plotter_update(i, x_getter, y_getters)

        json_path = os.path.join(os.path.dirname(self.load_path),"eval_buf.json")
        json.dump(self.buf, indent=2, fp=open(json_path, 'w'))
        print(f"Log data saved to {json_path}")

        fig_path = os.path.join(os.path.dirname(self.load_path),"eval_rewards.png")
        self.save_plot(fig_path, y_getters)
        print(f"Plots saved to {fig_path}")

        # plot and save robot motion
        urdf_path = self.env_cfg.asset.file.format(ROOT_DIR=ROOT_DIR)
        robot = URDF.load(urdf_path)

        joint_angles = self.plot_buf['joint_angles']
        base_pos = self.plot_buf['base_pos']
        base_quat = self.plot_buf['base_quat']
        ref_joint_angles = self.plot_buf['ref_joint_angles']
        ref_base_pos = self.plot_buf['ref_base_pos']
        ref_base_quat = self.plot_buf['ref_base_quat']

        anim_path = os.path.join(os.path.dirname(self.load_path),"animation.mp4")
        animate_robot(anim_path, robot, base_pos, base_quat, joint_angles, ref_base_pos, ref_base_quat,
                      ref_joint_angles)
        print(f"Animation saved to {anim_path}")

    def save_plot(self, save_path, y_getters=None):
        for subplot_id in range(len(y_getters)):
            curr_plotter_ax = self.env.plotter_axs[subplot_id] if len(y_getters) > 1 else self.env.plotter_axs
            for line in self.env.plotter_lines[subplot_id]:
                curr_plotter_ax.add_line(line)
        self.env.plotter_fig.savefig(save_path)


def update(frame_id, ax1, ax2, robot, base_pos, base_quat, joint_traj, ref_base_pos, ref_base_quat, ref_joint_traj):
    frames = forward_kinematics(robot, base_pos[frame_id], base_quat[frame_id], joint_traj[frame_id])
    ref_frames = forward_kinematics(robot, ref_base_pos[frame_id], ref_base_quat[frame_id], ref_joint_traj[frame_id])
    plot_robot(ax1, robot, frames)
    plot_robot(ax2, robot, ref_frames, 'ro-', 'reference')


def animate_robot(save_path, robot, base_pos, base_quat, joint_traj, ref_base_pos, ref_base_quat, ref_joint_traj):
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax2= fig.add_subplot(122, projection='3d')
    ani = FuncAnimation(fig, update, frames=len(joint_traj),
                        fargs=(ax1, ax2, robot, base_pos, base_quat, joint_traj, ref_base_pos, ref_base_quat, ref_joint_traj),
                        interval=20)
    FFWriter = FFMpegWriter(fps=50)
    ani.save(save_path, writer=FFWriter)


if __name__ == '__main__':
    args = get_args()
    args.dv = False
    seed = 2  # TODO: you can change the seed to get a different behavior
    ip = Eval(args, seed)
    ip.play()
    ip.runner.close()
