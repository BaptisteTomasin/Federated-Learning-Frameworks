from paddle.fluid import optimizer, layers
from paddle_fl.paddle_fl.core.master.job_generator import JobGenerator
from paddle_fl.paddle_fl.core.strategy.fl_strategy_base import FLStrategyFactory
from tools import select_model
import json 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', help = "path to the config file")
args = parser.parse_args()

with open(args.config_path, 'r') as fp: 
    params = json.load(fp)

#  Define Model
###########################

inp = layers.data(name ='inp', shape = params["federated"]["input_shape"], dtype = params["federated"]["input_dtype"])
label = layers.data(name ='label', shape = params["federated"]["label_shape"], dtype = params["federated"]["label_dtype"])
if "number_of_class" in params["centralized"]:
    model = select_model(params["federated"]["model_name"], inp, label, params["federated"]["number_of_class"])
else:
    model = select_model(params["federated"]["model_name"], inp, label)

#  Clients configs
###########################
job_generator = JobGenerator()                       
optimizer = optimizer.SGD(learning_rate= params["federated"]["learning_rate"])
job_generator.set_optimizer(optimizer)               
job_generator.set_losses([model.loss])               
job_generator.set_startup_program(model.startup_program)
job_generator.set_infer_feed_and_target_names(
    [model.inputs.name, model.label.name],
    model.fetch_list)

#  Choose the Federated learning strategy
############################################
# FLStrategyFactory allow to choose between three strategies fed_avg, dpsgd, sec_agg
# I choose the fed_avg strategy

build_strategy = FLStrategyFactory()
build_strategy.fed_avg = True
build_strategy.inner_step = 1    

#  Create The federated learning strategy choosed
####################################################

strategy = build_strategy.create_fl_strategy()

# define Distributed-Config and generate fl_job
endpoints = ["127.0.0.1:8181"]
output = params["federated"]["job_path"]
job_generator.generate_fl_job(
    strategy, server_endpoints=endpoints, worker_num=params["federated"]["number_of_clients"], output=output)

print('finish!')
