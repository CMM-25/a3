import os
import random
import glob
import numpy as np

import torch
from animRL import ROOT_DIR


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if hasattr(attr, "__dict__"):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return


def update_cfgs_from_dict(env_cfg, train_cfg, update_cfg):
    env_cfg.rewards.terms = []
    update_class_from_dict(env_cfg, update_cfg["env_cfg"])
    update_class_from_dict(train_cfg, update_cfg["train_cfg"])


def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_load_path(root, load_run=-1, checkpoint=-1):
    try:
        runs = os.listdir(root)
        runs.sort()
        if 'exported' in runs: runs.remove('exported')
        if 'wandb' in runs: runs.remove('wandb')
        last_run = os.path.join(root, runs[-1])
    except:
        raise ValueError("No runs in this directory: " + root)
    if load_run == -1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)

    if checkpoint == -1:
        models = [file for file in os.listdir(load_run) if 'model' in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint)

    load_path = os.path.join(load_run, model)
    return load_path


def get_paths_from_pattern(files_pattern):
    if isinstance(files_pattern, list):
        files_paths = []
        for file_name in files_pattern:
            files_paths.extend(glob.glob(file_name.format(ROOT_DIR=ROOT_DIR)))
    else:
        files_paths = glob.glob(files_pattern.format(ROOT_DIR=ROOT_DIR))
    return sorted(files_paths)


def update_env_cfg_from_args(env_cfg, args):
    if args.dv is not None:
        env_cfg.viewer.enable_viewer = args.dv
    # if args.dr is not None:
    #     env_cfg.viewer.record_camera_imgs = args.dr
    if args.num_envs is not None:
        env_cfg.env.num_envs = args.num_envs
    if args.debug is not None:
        env_cfg.env.debug = args.debug
    return env_cfg


def update_train_cfg_from_args(train_cfg, args):
    if train_cfg is not None:
        if args.experiment_name is not None:
            train_cfg.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            train_cfg.runner.run_name = args.run_name
        if args.load_run is not None:
            train_cfg.runner.load_run = args.load_run
        if args.checkpoint is not None:
            train_cfg.runner.checkpoint = args.checkpoint
        if args.wb is not None:
            train_cfg.runner.wandb = args.wb
        if args.dr is not None:
            train_cfg.runner.record_gif = args.dr
    return train_cfg
