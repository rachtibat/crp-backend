
# CELERY
broker_url = 'amqp://localhost//' #production:password@localhost//'
result_backend = 'rpc://'
celery_node_name = "debug"
task_time_limit = 600
worker_max_tasks_per_child = 200

result_serializer = 'pickle'
task_serializer = 'pickle'
accept_content = ["json", "pickle"]

# FLASK
flask_port = 5050

# CRP
crp_batch_size= 16

