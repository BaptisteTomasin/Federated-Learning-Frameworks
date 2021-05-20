from paddle.fluid import optimizer
from paddle_fl.paddle_fl.core.master.job_generator import JobGenerator
from paddle_fl.paddle_fl.core.strategy.fl_strategy_base import FLStrategyFactory
from models import *

#  Define Model
###########################

model = Multilayer_perceptron()
model.lr_network()

#  Clients configs
###########################
job_generator = JobGenerator()                       
optimizer = optimizer.SGD(learning_rate=0.1)
job_generator.set_optimizer(optimizer)               
job_generator.set_losses([model.loss])               
job_generator.set_startup_program(model.startup_program)
job_generator.set_infer_feed_and_target_names(
    [model.inputs.name, model.label.name],
    [model.loss.name, model.accuracy.name])

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
output = "fl_job_config"
job_generator.generate_fl_job(
    strategy, server_endpoints=endpoints, worker_num=10, output=output)

print('finish!')
