import json

import matplotlib.pyplot as plt

from animRL import ROOT_DIR
from animRL.rewards.rewards import REWARDS
from animRL.dataloader.motion_loader import MotionLoader
from animRL.cfg.mimic.walk_config import WalkCfg, WalkTrainCfg
from animRL.cfg.mimic.cartwheel_config import CartwheelCfg, CartwheelTrainCfg
from animRL.utils.math import *
import torch


if __name__ == '__main__':
    device = 'cpu'

    task_name = 'walk' # walk or cartwheel
    test_name = 'policy' # random or policy
    cfg = WalkCfg()

    dt = 0.02
    num_joints = cfg.env.num_actions
    num_ee = len(cfg.asset.ee_offsets)
    num_envs = cfg.env.num_envs
    motion_loader = MotionLoader(device, cfg.motion_loader, dt, num_joints, num_ee)

    test_file = f"{ROOT_DIR}/resources/tests/test_{task_name}_{test_name}.json"
    with open(test_file, 'r') as f:
        data = json.load(f)
    for data_t in data:
        for x in data_t.keys():
            data_t[x] = torch.asarray(data_t[x], device=device)
        data_t['motion_loader'] = motion_loader


    rewards = REWARDS()
    tolerance = 0.0

    plot_data = {"base_height_rew": [],
                 "base_orientation_rew": [],
                 "joint_pos_rew":[],
                 "base_vel_rew":[],
                 "ee_pos_rew":[],
                 "joint_target_rate_rew": []
                 }
    reward_keys = list(plot_data.keys())

    i = 0
    for data_t in data:
        print(f'----- time step {i} ----')
        reward_base_height = rewards.reward_track_base_height(data_t, cfg.rewards.terms.track_base_height[0], tolerance)
        print("reward_base_height: ", reward_base_height)
        plot_data["base_height_rew"].append(reward_base_height.numpy())

        reward_base_ori = rewards.reward_track_base_orientation(data_t, cfg.rewards.terms.track_base_orientation[0], tolerance)
        print("reward_base_ori: ", reward_base_ori)
        plot_data["base_orientation_rew"].append(reward_base_ori.numpy())

        reward_joint_pos = rewards.reward_track_joint_pos(data_t, cfg.rewards.terms.track_joint_pos[0], tolerance)
        print("reward_joint_pos: ", reward_joint_pos)
        plot_data["joint_pos_rew"].append(reward_joint_pos.numpy())

        reward_base_vel = rewards.reward_track_base_vel(data_t, cfg.rewards.terms.track_base_vel[0], tolerance)
        print("reward_base_vel: ", reward_base_vel)
        plot_data["base_vel_rew"].append(reward_base_vel.numpy())

        reward_ee_pos = rewards.reward_track_ee_pos(data_t, cfg.rewards.terms.track_ee_pos[0], tolerance)
        print("reward_ee_pos: ", reward_ee_pos)
        plot_data["ee_pos_rew"].append(reward_ee_pos.numpy())

        reward_joint_target_rate = rewards.reward_joint_targets_rate(data_t, cfg.rewards.terms.joint_targets_rate[0], tolerance)
        print("reward_joint_target_rate: ", reward_joint_target_rate)
        plot_data["joint_target_rate_rew"].append(reward_joint_target_rate.numpy())

    # plot the rewards
    num_rewards = len(reward_keys)
    fig, axs = plt.subplots(num_rewards, 1, figsize=(12, 3 * num_rewards), sharex=True)

    for i, key in enumerate(reward_keys):
        ax = axs[i]
        for env_idx in range(4):
            sequence = np.array(plot_data[key])[:, env_idx]
            ax.plot(sequence, label=f"Env {env_idx + 1}")
        ax.set_title(key.replace("_", " ").title())
        ax.set_ylabel("Reward")
        ax.legend()
        ax.grid(True)

    axs[-1].set_xlabel("Timestep")
    plt.tight_layout()
    plt.show()