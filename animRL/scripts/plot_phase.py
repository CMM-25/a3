# created by Fatemeh Zargarbashi - 2025
import isaacgym

from animRL import ROOT_DIR
from animRL.utils.isaac_helpers import get_args
from animRL.envs import task_registry

import torch
import matplotlib.pyplot as plt


class Eval:

    def __init__(self, args, seed=1):
        self.args = args

        env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

        # overwrite config params
        env_cfg.seed = seed
        env_cfg.env.num_envs = 4

        env_cfg.env.play = True
        env_cfg.sim.use_gpu_pipeline = False

        self.env_cfg = env_cfg
        self.train_cfg = train_cfg

        # prepare environment, runner and policy
        env, _ = task_registry.make_env(name=self.args.task, args=self.args, env_cfg=self.env_cfg)
        self.env = env
        self.runner = task_registry.make_alg_runner(env=env, name=self.args.task, args=self.args, env_cfg=self.env_cfg,
                                                    train_cfg=self.train_cfg)

    def play(self):
        # rollout
        num_steps = 200
        self.phase_buf = torch.zeros(num_steps, self.env.num_envs, device=self.env.device)
        self.env.reset()

        for i in range(num_steps):
            self.phase_buf[i] = self.env.phase.clone()
            actions = torch.zeros_like(self.env.actions)
            self.obs, _, _, self.dones, self.info = self.env.step(actions.detach())

        phase_data = self.phase_buf.cpu().numpy()

        # Plot
        plt.figure(figsize=(10, 5))
        for env_idx in range(phase_data.shape[1]):
            plt.plot(phase_data[:, env_idx], label=f'Env {env_idx + 1}')

        plt.title("Phase Over Time")
        plt.xlabel("Timestep")
        plt.ylabel("Phase")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(f'{ROOT_DIR}/results/phase_plot.png')


if __name__ == '__main__':
    args = get_args()
    args.dv = False
    seed = 1
    ip = Eval(args, seed)
    ip.play()
    ip.runner.close()
