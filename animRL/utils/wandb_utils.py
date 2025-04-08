import os

from torch.utils.tensorboard import SummaryWriter
from animRL.utils.helpers import class_to_dict

try:
    import wandb
except ModuleNotFoundError:
    raise ModuleNotFoundError("Wandb is required to log to Weights and Biases.")


class WandbSummaryWriter(SummaryWriter):
    def __init__(self, log_dir: str, flush_secs: int, cfg, group=None):
        super().__init__(log_dir, flush_secs)

        try:
            project = cfg.experiment_name
        except KeyError:
            raise KeyError(
                "Please specify wandb_project in the runner config."
            )

        project_log_path = os.path.dirname(os.path.abspath(log_dir))
        run_name = os.path.basename(os.path.abspath(log_dir))
        wandb.init(project=project, dir=project_log_path, group=group)
        wandb.run.name = run_name + "-" + wandb.run.name.split("-")[-1]

        self.log_dict = {}

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, new_style=False):
        super().add_scalar(
            tag,
            scalar_value,
            global_step=global_step,
            walltime=walltime,
            new_style=new_style,
        )
        self.log_dict[tag] = scalar_value

    def flush_logger(self, global_step):
        wandb.log(self.log_dict, step=global_step)
        self.log_dict = {}

    def stop(self):
        wandb.finish()

    def log_config(self, env_cfg, runner_cfg, algorithm_cfg, policy_cfg, alg_name=None):
        if alg_name is not None:
            wandb.config.update({"algorithm_name": alg_name})
        wandb.config.update({"runner_cfg": class_to_dict(runner_cfg)})
        wandb.config.update({"policy_cfg": class_to_dict(policy_cfg)})
        wandb.config.update({"algorithm_cfg": class_to_dict(algorithm_cfg)})
        wandb.config.update({"env_cfg": class_to_dict(env_cfg)})
