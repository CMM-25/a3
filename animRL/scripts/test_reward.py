import json

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
    cfg = WalkCfg()

    dt = 0.02
    num_joints = cfg.env.num_actions
    num_ee = len(cfg.asset.ee_offsets)
    num_envs = cfg.env.num_envs
    motion_loader = MotionLoader(device, cfg.motion_loader, dt, num_joints, num_ee)

    test_file = f"{ROOT_DIR}/resources/tests/test2.json"
    with open(test_file, 'r') as f:
        data = json.load(f)
    for x in data.keys():
        data[x] = torch.asarray(data[x], device=device)

    data['motion_loader'] = motion_loader


    rewards = REWARDS()
    tolerance = 0.0

    reward_base_height = rewards.reward_track_base_height(data, cfg.rewards.terms.track_base_height[0], tolerance)
    print(reward_base_height)

    reward_base_ori = rewards.reward_track_base_orientation(data, cfg.rewards.terms.track_base_orientation[0], tolerance)
    print(reward_base_ori)

    reward_joint_pos = rewards.reward_track_joint_pos(data, cfg.rewards.terms.track_joint_pos[0], tolerance)
    print(reward_joint_pos)

    reward_base_vel = rewards.reward_track_base_vel(data, cfg.rewards.terms.track_base_vel[0], tolerance)
    print(reward_base_vel)

    reward_ee_pos = rewards.reward_track_ee_pos(data, cfg.rewards.terms.track_ee_pos[0], tolerance)
    print(reward_ee_pos)

    reward_joint_target_rate = rewards.reward_joint_targets_rate(data, cfg.rewards.terms.joint_targets_rate[0], tolerance)
    print(reward_joint_target_rate)
