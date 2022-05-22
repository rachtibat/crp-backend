
import warnings
warnings.filterwarnings("ignore", message="torch.distributed.reduce_op")

from server.flask_server import run_socket_flask
from CRP_backend.server.celery_tasks import celapp
import CRP_backend.server.config as config

import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)


def start_celery_worker(hostname, queue, pool, concurrency=1):

    worker = celapp.Worker(hostname=hostname, pool=pool, loglevel="INFO", queues=[queue], concurrency=concurrency)
    worker.start()


def set_config():
    pass
    #TODO:fill as option in __main__ which modifies the celery_config.py somehow as json as example..

if __name__ == "__main__":

    node = config.celery_node_name
    experiments = {"VGG16_ImageNet": "cuda:0"} #"LeNet_Fashion": "cuda:0", 
    
    ######## start GPU workers
    if "celery" in experiments:
        raise ValueError("The name 'celery' is reserved. Please rename your experiment.")

    processes = []
    for i, e_name in enumerate(experiments):

        p = mp.Process(target=start_celery_worker, args=(f"worker_{i}@{node}", e_name, "solo"))
        p.start()
        processes.append(p)

    ######## start IO workers
    #TODO: combine eventlet with prefork by spawning this in for loop with eventlet
    n_processes = 4
    p = mp.Process(target=start_celery_worker, args=(f"worker_IO@{node}", "celery", "prefork", n_processes))
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