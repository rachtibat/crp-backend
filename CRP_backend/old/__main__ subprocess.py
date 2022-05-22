from celery.bin import worker
from server.flask_server import run_socket_flask
from CRP_backend.server.data_tasks import celapp
import subprocess
from pathlib import Path

import multiprocessing as mp

#bash_command = "celery -A CRP_backend.server.celery_server worker -P solo"
#cwd_path = Path(__file__).parent
#print(cwd_path)

#process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE, cwd=cwd_path)
#output, error = process.communicate()
#print(output, error)

run_socket_flask(debug=False)

worker = celapp.Worker()
worker.start()