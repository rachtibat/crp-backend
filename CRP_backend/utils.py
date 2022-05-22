
import os
import importlib.util

from CRP_backend.interface import Interface


def get_interface(name, path=None) -> Interface:

    if path is None:
        path = os.getcwd()
  
    exp_names = os.listdir(path)

    if name not in exp_names:
        raise FileNotFoundError(f"Experiment {name} not found in {path}")

    file_path = path + f"/{name}/" + "crp_interface.py"

    try:

        spec = importlib.util.spec_from_file_location("CRP_Interface", file_path)
        myModule = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(myModule)

        Interface =  myModule.CRP_Interface()
         
    except Exception as e:
        print(e)
        raise ImportError(
            f"Could not load file <CRP_Interface> from {file_path}")

    return Interface