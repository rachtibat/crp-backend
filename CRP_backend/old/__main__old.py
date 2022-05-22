from core.Experiment import load_model_and_data, load_model_graph
from datatypes.test_suite import TestSuite
from interface.main_utils import *
from interface.questions import *
import argparse
import time
from sys import exit

from feature_visualization.MaxActivation_old import MaxActivation
from feature_visualization.ReceptiveField import ReceptiveField
from feature_visualization.ConceptAnalysis import ConceptAnalysis
from zennit_API.API import ZennitAPI

from server.socket_backend import METHODS

parser = argparse.ArgumentParser(description='XAI tool.')
parser.add_argument('experiment', help='experiment name.', type=str, nargs="+")
group = parser.add_mutually_exclusive_group()
group.add_argument(
    '-t', '--test', help='run test of model and data.', action='store_true')
group.add_argument('-arg', "--argfile",
                   help='generate argfile', action='store_true')
group.add_argument('-sub', "--submit",
                   help='submit argfile line', action='store_true')
group.add_argument('-server', "--server",
                   help='start server', action='store_true')
parser.add_argument(
    '-d', '--device', help='device on which python data and model is loaded', type=str)
parse_extra_analysis_arguments(parser)

args = parser.parse_args()

if len(args.experiment) > 1:
    print("Please select only one experiment for testing, generating or submitting argfile.txt")
    exit()

start_time = time.time()
device = select_torch_device(args.device)
config_message = load_config(args.experiment[0])

SDS = load_model_and_data(args.experiment[0], device, config_message)

# RUN TESTS
if args.test:
    TS = TestSuite(SDS)
    TS.run_all_tests()

    MG = load_model_graph(SDS.DMI)
    #TS.generate_synthetic_images(args.experiment[0], MG)
    exit()


MG = load_model_graph(SDS.DMI)

ZAPI = ZennitAPI(SDS.DMI, MG, device)

save_path = load_save_path(args.experiment[0], SDS.DMI)

# GENERATE ARGFILE
if args.argfile:

    argfile_arguments = []

    save_results_question(save_path)
    analysis_answer = analysis_question()

    print("Calculate argfile.txt...")

    if 'Concept Analysis' in analysis_answer:

        n_processes = processes_question('Concept Analysis')

        #if 'Max Relevance Inter Class' in analysis_answer:
        method = method_question(METHODS)

        CA = ConceptAnalysis(MG, SDS, ZAPI, save_path, analysis_answer, config_message)
        command_args = CA.divide_processes(int(n_processes))

        analysis_string = "A"
        if 'Max Activation Inter Class' in analysis_answer:
            analysis_string += "AC"
        if 'Max Relevance Inter Class' in analysis_answer:
            analysis_string += "RC"

        for arg in command_args:
            argfile_arguments.append(
                f"--CA " + arg + " " + method + " " + analysis_string)

    if 'Receptive Field of Filters' in analysis_answer:

        n_processes = processes_question('Receptive Field of Filters')

        RF = ReceptiveField(args.experiment[0], MG, SDS.DMI, ZAPI, save_path)
        print("One moment please ...")
        command_args = RF.divide_processes(int(n_processes))

        for arg in command_args:
            argfile_arguments.append(f"--RF " + arg)

    print(" finished")
    save_argfile(argfile_arguments, args.experiment[0])


# SUBMIT command line argument:
if args.submit:

    # Concept Analysis
    if args.CA:
        arg = args.CA
        modes = []
        print(
            f"Load ConceptAnalysis module and start analysis with args: {arg}...")
        if "AC" in arg[3]:
            modes.append('Max Activation Inter Class')
        if "RC" in arg[3]:
            modes.append('Max Relevance Inter Class')

        CA = ConceptAnalysis(MG, SDS, ZAPI, save_path, modes, config_message)
        CA.run_analysis(int(arg[0]), int(arg[1]), arg[2], BATCH_SIZE=16)
        print("finished.")
        command_args = load_argfile(args.experiment[0])["CA"]
        CA.collect_results(command_args)

    # Receptive Field Analysis
    if args.RF:
        arg = args.RF
        print(
            f"Load ReceptiveField module and start analysis with args: {arg}...")
        RF = ReceptiveField(args.experiment[0], MG, SDS.DMI, ZAPI, save_path)
        RF.run_analysis(arg[0], arg[1], arg[2], arg[3], BATCH_SIZE=16)
        print("finished.")
        command_args = load_argfile(args.experiment[0])["RF"]
        RF.collect_results(command_args)


print(f"time elapsed: {time.time()-start_time : .2f} s")