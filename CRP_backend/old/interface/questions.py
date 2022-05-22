from InquirerPy import prompt
import sys
from pathlib import Path
import os
import json

from CRP_backend.core.Experiment import load_config

def experiment_select_question(names):
    selected = [
        {
            'type': 'checkbox',
            'message': 'Select Several Experiments:',
            'name': 'selected',
            'choices': [],
            'validate': lambda answer: 'You must choose at least one.' \
                if len(answer) == 0 else True
        }
    ]

    for name in names:
        selected[0]["choices"].append(name)

    answer = prompt(selected)["selected"]

    if len(answer) == 0:
        print("Please choose one option with <space> key.")
        sys.exit()

    return answer

def analysis_question():
    """
    Ask user which analysis should be performed
    """

    analysis_type = [
        {
            'type': 'checkbox',
            'message': 'Select calculations to perform:',
            'name': 'analysis',
            'choices': [
                {
                    'name': 'Max Relevance w.r.t Class Statistics', 'value': 'Max Relevance Inter Class', 'enabled': False,
                },
                {
                    'name': 'Max Activation w.r.t Class Statistics', 'value': 'Max Activation Inter Class', 'enabled': False,
                },
                {
                    'name': 'Max Activation (mandatory)', 'value': 'Max Activation', 'enabled': False,
                },
                {
                    'name': 'Receptive Field of Filters (mandatory)', 'value': 'Receptive Field of Filters', 'enabled': False,
                }

            ],
            'validate': lambda answer: 'You must choose at least one.' \
                if len(answer) == 0 else True
        }
    ]

    analysis_answer = prompt(analysis_type)["analysis"]

    if len(analysis_answer) == 0:
        print("Please choose one option with <space> key.")
        sys.exit()

    # summarize MaxAct, MaxRel as "Concept Analysis" to make it easier to handle
    if not(len(analysis_answer) == 1 and 'Receptive Field of Filters' in analysis_answer):
        analysis_answer.append("Concept Analysis")

    return analysis_answer


def processes_question(analysis):
    processQuantityQuestion = {
        'type': 'input',
        'message': f'How many processes to spawn maximally for {analysis}?',
        'name': 'n_processes',
        'default': "1",
        'validate': lambda val: True if int(val) > 0 else "Choose a value greater than 0"
    }

    n_processes = prompt(processQuantityQuestion)["n_processes"]

    return n_processes


def save_results_question(save_path):
    """
        Confirm save path defined in config file.
    """

    save_question = [{
        'type': 'confirm',
        'message': f'Save calculations in: {save_path}',
        'name': 'path',
        'default': True,
    }]

    answer = prompt(save_question)
    if not answer["path"]:
        print("Please define path as <save_path> key in the config.json file.")
        sys.exit()


def method_question(METHODS):

    selected = [
        {
            'type': 'list',
            'message': 'Select Zennit Method for Relevance Analysis:',
            'name': 'selected',
            'choices': [],
            'validate': lambda answer: 'You must choose one.' \
                if len(answer) == 0 else True
        }
    ]

    for m in METHODS:
        selected[0]["choices"].append(m)

    answer = prompt(selected)["selected"]

    if len(answer) == 0:
        print("Please choose one option with <space> key.")
        sys.exit()

    return answer


