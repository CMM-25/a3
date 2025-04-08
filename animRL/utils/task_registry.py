import json
import os
from datetime import datetime

from animRL.utils.isaac_helpers import get_args, parse_sim_params
from animRL.utils.helpers import update_env_cfg_from_args, update_train_cfg_from_args, class_to_dict, get_load_path, \
    set_seed

from animRL import ROOT_DIR
from animRL.runners.algorithms.ppo import PPO


class TaskRegistry():
    def __init__(self):
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}

    def register(self, name: str, task_class, env_cfg, train_cfg):
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg

    def get_task_class(self, name: str):
        return self.task_classes[name]

    def get_cfgs(self, name):
        train_cfg = self.train_cfgs[name]
        env_cfg = self.env_cfgs[name]
        return env_cfg, train_cfg

    def make_env(self, name, args=None, env_cfg=None):
        if args is None:
            args = get_args()
        # check if there is a registered env with that name
        if name in self.task_classes:
            task_class = self.get_task_class(name)
        else:
            raise ValueError(f"Task with name: {name} was not registered")
        if env_cfg is None:
            # load config files
            env_cfg, _ = self.get_cfgs(name)
        # override cfg from args (if specified)
        env_cfg = update_env_cfg_from_args(env_cfg, args)
        set_seed(env_cfg.seed)
        # parse sim params (convert to dict first)
        sim_params = {"sim": class_to_dict(env_cfg.sim)}
        sim_params = parse_sim_params(args, sim_params)
        env = task_class(cfg=env_cfg,
                         sim_params=sim_params,
                         physics_engine=args.physics_engine,
                         sim_device=args.sim_device,
                         headless=args.headless)
        return env, env_cfg

    def make_alg_runner(self, env, name=None, args=None, env_cfg=None, train_cfg=None):
        if args is None:
            args = get_args()
        # if config files are passed use them, otherwise load from the name
        create_and_save = False
        if train_cfg is None:
            _, train_cfg = self.get_cfgs(name)
            create_and_save = not args.debug
        train_cfg = update_train_cfg_from_args(train_cfg, args)
        log_root = os.path.join(ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
        log_dir = os.path.join(log_root, datetime.now().strftime('%Y-%m-%d_%H%M%S') + '_' + train_cfg.runner.run_name)

        if create_and_save:
            os.makedirs(log_dir)
            env_cfg_dict = class_to_dict(env_cfg)
            train_cfg_dict = class_to_dict(train_cfg)
            env_cfg_dict["viewer"]["enable_viewer"] = False
            cfg = {
                "train_cfg": train_cfg_dict,
                "env_cfg": env_cfg_dict
            }

            with open(os.path.join(log_dir + '/' + train_cfg.runner.experiment_name + '.json'), 'w') as f:
                json.dump(cfg, f, indent=2)

        algorithm = eval(train_cfg.algorithm_name)
        runner = algorithm(
            env=env,
            train_cfg=train_cfg,
            log_dir=log_dir,
            device=args.device)

        # # save resume path before creating a new log_dir
        # resume = train_cfg.runner.resume
        # if resume:
        #     # load previously trained model
        #     resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run,
        #                                 checkpoint=train_cfg.runner.checkpoint)
        #     print(f"Loading model from: {resume_path}")
        #     runner.load(resume_path)
        return runner


# make global task registry
task_registry = TaskRegistry()
