import os
import CRP_backend
import pickle
import numpy as np

e_path = os.path.dirname(CRP_backend.__file__) + "/experiments"


def load_max_activation(experiment, layer_name):
    t_path = e_path + f"/{experiment}/"

    with open(t_path + f"MaxActivation/{layer_name}_a_value.p", 'rb') as fp:
        most_act_value = pickle.load(fp)

    with open(t_path + f"MaxActivation/{layer_name}_d_index.p", 'rb') as fp:
        most_data_index = pickle.load(fp)

    with open(t_path + f"MaxActivation/{layer_name}_n_index.p", 'rb') as fp:
        most_neuron_index = pickle.load(fp)

    return most_act_value, most_data_index, most_neuron_index


def load_max_relevance(experiment, layer_name):
    t_path = e_path + f"/{experiment}/"

    with open(t_path + f"MaxRelevance/{layer_name}_a_value.p", 'rb') as fp:
        most_act_value = pickle.load(fp)

    with open(t_path + f"MaxRelevance/{layer_name}_d_index.p", 'rb') as fp:
        most_data_index = pickle.load(fp)

    with open(t_path + f"MaxRelevance/{layer_name}_n_index.p", 'rb') as fp:
        most_neuron_index = pickle.load(fp)

    return most_act_value, most_data_index, most_neuron_index


def load_receptive_field(experiment, layer_name):
    t_path = e_path + f"/{experiment}/"

    rf = np.load(t_path + f"ReceptiveField/layer_{layer_name}.npy", mmap_mode='r')

    #with open(t_path + f"ReceptiveField/layer_{layer_name}.p", 'rb') as fp:
     #   rf = pickle.load(fp)

    #rf_to_neuron_index = np.load(t_path + f"ReceptiveField/layer_{layer_name}_indices.npy", mmap_mode='r')

    with open(t_path + f"ReceptiveField/layer_{layer_name}_indices.p", 'rb') as fp:
        rf_to_neuron_index = pickle.load(fp)

    return rf, rf_to_neuron_index
    

def load_rel_statistics(experiment, layer_name):

    t_path = e_path + f"/{experiment}/"

    with open(t_path + f"MaxRelevanceInterClass/{layer_name}_r_value.p", 'rb') as fp:
        most_rel_value = pickle.load(fp)

    with open(t_path + f"MaxRelevanceInterClass/{layer_name}_d_index.p", 'rb') as fp:
        most_data_index = pickle.load(fp)

    with open(t_path + f"MaxRelevanceInterClass/{layer_name}_n_index.p", 'rb') as fp:
        most_neuron_index = pickle.load(fp)

    with open(t_path + f"MaxRelevanceInterClass/{layer_name}_mean.p", 'rb') as fp:
        mean_rl = pickle.load(fp)

    return most_rel_value, most_data_index, most_neuron_index, mean_rl


def load_act_statistics(experiment, layer_name):

    t_path = e_path + f"/{experiment}/"

    with open(t_path + f"MaxActivationInterClass/{layer_name}_r_value.p", 'rb') as fp:
        most_rel_value = pickle.load(fp)

    with open(t_path + f"MaxActivationInterClass/{layer_name}_d_index.p", 'rb') as fp:
        most_data_index = pickle.load(fp)

    with open(t_path + f"MaxActivationInterClass/{layer_name}_n_index.p", 'rb') as fp:
        most_neuron_index = pickle.load(fp)

    with open(t_path + f"MaxActivationInterClass/{layer_name}_mean.p", 'rb') as fp:
        mean_rl = pickle.load(fp)

    return most_rel_value, most_data_index, most_neuron_index, mean_rl


