
import warnings
warnings.filterwarnings("ignore", message="torch.distributed.reduce_op")

from server.flask_server import run_socket_flask
from CRP_backend.server.celery_tasks import celapp
import CRP_backend.server.config as config

import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)


def start_celery_worker(hostname, experiment, device, pool, concurrency=1):

    if "cuda" in device:
        queue = experiment + "_cuda"
    elif "cpu" in device:
        queue = experiment + "_cpu"
    else:
        raise ValueError("'device' must contain 'cuda' or 'cpu'")

    worker = celapp.Worker(hostname=hostname, pool=pool, loglevel="INFO", queues=[queue], concurrency=concurrency, experiment=experiment, device=device)
    worker.start()


def set_config():
    pass
    #TODO:fill as option in __main__ which modifies the celery_config.py somehow as json as example..

if __name__ == "__main__":

    n_processes = 4

    node = config.celery_node_name
    experiments = {"VGG16_bn_ImageNet": "cuda:0"} #"LeNet_Fashion": "cuda:0", 
    
    if "celery" in experiments:
        raise ValueError("The name 'celery' is reserved. Please rename your experiment.")

    processes = []
    for e_name, device in experiments.items():

        ######## start GPU workers
        p = mp.Process(target=start_celery_worker, args=(f"{device}_worker_{e_name}@{node}", e_name, device, "solo"))
        p.start()
        processes.append(p)

        ######## start CPU workers
        #TODO: combine eventlet with prefork by spawning this in for loop with eventlet
        p = mp.Process(target=start_celery_worker, args=(f"cpu_worker_{e_name}@{node}", e_name, "cpu", "prefork", n_processes))
        p.start()
        processes.append(p)

    ######## flask
    p = mp.Process(target=run_socket_flask, args=(experiments, "0.0.0.0", config.flask_port))
    p.start()
    processes.append(p)

    ######## end
    for p in processes:
        p.join()

    for i in range(10):
        print("should not print (:")