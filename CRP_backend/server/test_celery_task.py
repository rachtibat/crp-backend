
from celery import chain
from CRP_backend.server.celery_tasks import *

experiment = "VGG16_ImageNet"
concept_id = 0
l_name = "features.40"
mode = "relevance"
r_range = (0, 8)
plot_mode = "image and heatmap"
method = "epsilon_plus_flat"
rf = True
fn_name = "get_max_reference"
job = "test"
request_sid = 0
size = 224
device = "cuda:0"

#r = load_cache(experiment, concept_id, l_name, mode, r_range, method, rf, fn_name, plot_mode)
   
r = get_max_reference(experiment, device, concept_id, l_name, mode, r_range, method, rf, plot_mode)
    
save_cache(r, experiment, l_name, mode, r_range, method, rf, fn_name, plot_mode)
send_reference(r, job, experiment, request_sid, concept_id, l_name, mode, fn_name, plot_mode, size, None)

