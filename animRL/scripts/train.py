import isaacgym # keep this import here
from animRL.envs import *
from animRL.utils.task_registry import task_registry
from animRL.utils.isaac_helpers import get_args


def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    runner = task_registry.make_alg_runner(env=env, name=args.task, args=args, env_cfg=env_cfg)
    runner.learn()
    runner.close()


if __name__ == '__main__':
    args = get_args()
    train(args)
