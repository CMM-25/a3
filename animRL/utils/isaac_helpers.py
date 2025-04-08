from isaacgym import gymapi
from isaacgym import gymutil


def parse_sim_params(args, cfg):
    # code from Isaac Gym Preview 2
    # initialize sim params
    sim_params = gymapi.SimParams()

    # set some values from args
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params


def get_args():
    custom_parameters = [
        {
            "name": "--task",
            "type": str,
            "default": "walk",
            "help": "Name of the task to run."
        },
        {
            "name": "--experiment_name",
            "type": str,
            "help": "Name of the experiment to run or load. Overrides config file if provided."
        },
        {
            "name": "--run_name",
            "type": str,
            "help": "Name of the run. Overrides config file if provided."
        },
        {
            "name": "--load_run",
            "type": str,
            "default": -1,
            "help": "Name of the run to load when resume=True. If -1: will load the last run. "},
        {
            "name": "--checkpoint",
            "type": int,
            "default": -1,
            "help": "Saved model checkpoint number. If -1: will load the last checkpoint. "},
        {
            "name": "--device",
            "type": str,
            "default": "cuda:0",
            "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {
            "name": "--dv",
            "action": "store_false",
            "default": True,
            "help": "Disable viewer",
        },
        {
            "name": "--wb",
            "action": "store_true",
            "default": False,
            "help": "Turn on Weights and Bias writer",
        },
        {
            "name": "--dr",
            "action": "store_false",
            "default": True,
            "help": "Disable recording gifs",
        },
        {
            "name": "--num_envs",
            "type": int,
            "help": "Number of environments to create. Overrides config file if provided."
        },
        {
            "name": "--debug",
            "action": "store_true",
            "default": False, "help": "Debug mode to disable logging"
        },
    ]
    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)

    # name alignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    args.headless = not args.dv
    if args.sim_device == 'cuda':
        args.sim_device += f":{args.sim_device_id}"
    return args
