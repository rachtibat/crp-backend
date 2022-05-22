from CRP_backend.core.Experiment import load_config
import CRP_backend
import torch
import os

e_path = os.path.dirname(CRP_backend.__file__) + "/experiments"


def show_experiments_dictionaries():
    all_names = os.listdir(e_path)

    return all_names

def select_torch_device(name) -> torch.device:

    if name:
        all_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] + ["cpu"]
        if name not in all_devices:
            raise KeyError(f"Device {name} not available. Available are: {all_devices}")
        device = torch.device(name)
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

    return device


def load_save_path(experiment, DMI):

    config_file = load_config(experiment)

    save_path = config_file["save_path"]

    if not save_path:
        save_path = DMI.file_path

    return save_path


def parse_extra_analysis_arguments(parser):

    parser.add_argument('--EXP', metavar=("E"), type=str, nargs=1,
                        help='Name of Experiment')

    parser.add_argument('--MA', metavar=("D", "D"), type=int, nargs=2,
                        help='Data index start, Data index stop exclusive')
    
    parser.add_argument('--CA', metavar=("D", "D", "M", "A"), type=str, nargs=4,
                        help='Data index start, Data index stop exclusive, Zennit Method, Analysis Mode')

    parser.add_argument('--RF', metavar=("L", "N", "L", "N"), type=int, nargs=4,
                        help='Layer index start, Neuron index start, Layer index stop, Neuron index stop exclusive')


def load_argfile(experiment: str):

    command_args = { "MA" : [], "RF" : [], "CA" : []}
    t_path = f"{e_path}/{experiment}/argfile.txt"
    with open(t_path, "r") as f:
        for line in f:
            arg = line.rstrip('\n')
            if "MA" in arg:
                command_args["MA"].append(arg)
            if "RF" in arg:
                command_args["RF"].append(arg)
            if "CA" in arg:
                command_args["CA"].append(arg)

    return command_args


def save_argfile(string_array: list, experiment):

    t_path = f"{e_path}/{experiment}/argfile.txt"
    with open(t_path, "w") as f:
        f.write("\n".join(string_array))

    print(f"argfile.txt saved in {t_path}")