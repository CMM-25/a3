from animRL.utils.task_registry import task_registry

from .mimic.mimic_task import MimicTask
from ..cfg.mimic.walk_config import WalkCfg, WalkTrainCfg
from ..cfg.mimic.cartwheel_config import CartwheelCfg, CartwheelTrainCfg


task_registry.register("walk", MimicTask, WalkCfg(), WalkTrainCfg())
task_registry.register("cartwheel", MimicTask, CartwheelCfg(), CartwheelTrainCfg())