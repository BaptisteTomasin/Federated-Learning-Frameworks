from paddle_fl.paddle_fl.core.trainer.fl_trainer import FLTrainerFactory
from paddle_fl.paddle_fl.core.master.fl_job import FLRunTimeJob
import paddle
from tb_paddle import SummaryWriter

from os.path import join 
import sys
import logging
import time
import json
import argparse

from tools import metrics, select_data

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', help = "path to the config file")
parser.add_argument('--id', help = "path to the config file")
args = parser.parse_args()

with open(args.config_path, 'r') as fp: 
    params = json.load(fp)

#  Log
#########

logging.basicConfig(
    filename="test.log",
    filemode="w",
    format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
    datefmt="%d-%M-%Y %H:%M:%S",
    level=logging.DEBUG)

#  Load configs
####################
trainer_id = int(args.id)  # trainer id
job_path = params["federated"]["job_path"]
job = FLRunTimeJob()
job.load_trainer_job(job_path, trainer_id)
job._scheduler_ep = "127.0.0.1:"+ str(params["federated"]["scheduler_port"])  # Inform scheduler IP address to trainer
trainer = FLTrainerFactory().create_fl_trainer(job)
trainer._current_ep = "127.0.0.1:{}".format(params["federated"]["seed_of_clients_port"] + trainer_id)
place = paddle.fluid.CPUPlace()
trainer.start(place)

test_program = trainer._main_program.clone(for_test = True)

#  Load data 
###############

# dataset = Time_series_loader(distributed = params["federated"]["distributed"], ts_path = params["federated"]["clients_path"], number_of_clients = params["federated"]["number_of_clients"], lookback = params["federated"]["lookback"], lookforward = params["federated"]["lookforward"])
dataset = select_data(params)

train_reader = paddle.batch(reader = dataset.train_data(client = trainer_id),
                            batch_size = params["federated"]["batch_size"])
val_reader = paddle.batch(reader=dataset.val_data(client = trainer_id), 
                            batch_size = params["federated"]["batch_size"])
if trainer_id == 0:
    test_reader = paddle.batch(reader=dataset.test_data(), 
                                batch_size = params["federated"]["batch_size"])


inp = paddle.fluid.layers.data(name ='inp', shape = params["federated"]["input_shape"], dtype = params["federated"]["input_dtype"])
label = paddle.fluid.layers.data(name ='label', shape = params["federated"]["label_shape"], dtype = params["federated"]["label_dtype"])
feeder = paddle.fluid.DataFeeder(feed_list = [inp, label], place = paddle.fluid.CPUPlace())

# Summary
###########
data_writer = SummaryWriter(logdir=join(join(params["federated"]["logdir"],"data"),f"client_{trainer_id}"))

#  Run
#########
round_id = 0
while not trainer.stop():
    round_id += 1

    if round_id > params["federated"]["num_round"]:
        break

    for e in range(params["federated"]["num_epoch"]):
        for data in train_reader():
            trainer.run(feeder.feed(data), fetch=job._target_names)
    
    train_metrics = metrics(trainer.exe, test_program,feeder, train_reader, job._target_names)
    val_metrics = metrics(trainer.exe, test_program,feeder, val_reader, job._target_names)
    if trainer_id == 0:
        test_metrics = metrics(trainer.exe, test_program,feeder, test_reader, job._target_names)


    txt_log = "{} Round {} ".format(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),
                                     round_id)

    for metric in range(len(job._target_names)):
        metric_name = job._target_names[metric]
        txt_log += f"Train {metric_name}: {train_metrics[metric]} Val {metric_name}: {val_metrics[metric]}"
        data_writer.add_scalar(f"train/{metric_name}", train_metrics[metric], round_id)
        data_writer.add_scalar(f"val/{metric_name}", val_metrics[metric], round_id)
        if trainer_id == 0:
            txt_log += f" Test {metric_name}: {test_metrics[metric]} "
            data_writer.add_scalar(f"test/{metric_name}", test_metrics[metric], round_id)
    
    print(txt_log)