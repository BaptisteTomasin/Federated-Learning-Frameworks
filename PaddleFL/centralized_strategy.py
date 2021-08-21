import paddle 
from tb_paddle import SummaryWriter

from os.path import join 
import time
import json
import argparse

from tools import metrics, select_model, select_data

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', help = "path to the config file")
args = parser.parse_args()

with open(args.config_path, 'r') as fp: 
    params = json.load(fp)

#  Load data 
###############
dataset = select_data(params)

train_reader = paddle.batch(reader = dataset.train_data(),
                            batch_size = params["centralized"]["batch_size"])
val_reader = paddle.batch(reader=dataset.val_data(), 
                            batch_size = params["centralized"]["batch_size"])
test_reader = paddle.batch(reader=dataset.test_data(), 
                            batch_size = params["centralized"]["batch_size"])

inp = paddle.fluid.layers.data(name ='inp', shape = params["centralized"]["input_shape"], dtype = params["centralized"]["input_dtype"])
label = paddle.fluid.layers.data(name ='label', shape = params["centralized"]["label_shape"], dtype = params["centralized"]["label_dtype"])
feeder = paddle.fluid.DataFeeder(feed_list = [inp, label], place = paddle.fluid.CPUPlace())

#  Define Model
#################
if "number_of_class" in params["centralized"]:
    model = select_model(params["centralized"]["model_name"], inp, label, params["centralized"]["number_of_class"])
else:
    model = select_model(params["centralized"]["model_name"], inp, label)

#   Train
###############
test_program = paddle.fluid.default_main_program().clone(for_test = True)

# Define optimization method
optimizer = paddle.fluid.optimizer.SGD(learning_rate = params["centralized"]["learning_rate"])
opts = optimizer.minimize(model.loss)

# Define a parser that uses the CPU
place = paddle.fluid.CPUPlace()
exe = paddle.fluid.Executor(place)
#Parameter initialization
exe.run(paddle.fluid.default_startup_program())

# Summary
###########
data_writer = SummaryWriter(logdir = join(params["centralized"]["logdir"],"data"))

#Start training and testing
for epoch in range(params["centralized"]["num_epoch"]):

    for data in train_reader():
        exe.run(program=paddle.fluid.default_main_program(),
                                        feed=feeder.feed(data),
                                        fetch_list=model.fetch_list) 

    train_metrics = metrics(exe, test_program,feeder, train_reader, model.fetch_list)
    val_metrics = metrics(exe, test_program,feeder, val_reader, model.fetch_list)
    test_metrics = metrics(exe, test_program,feeder, test_reader, model.fetch_list)


    txt_log = "{} Epoch {} ".format(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), epoch)
    for metric in range(len(model.fetch_list)):
        metric_name = model.fetch_list[metric]
        txt_log += f"Train {metric_name}: {train_metrics[metric]} Val {metric_name}: {val_metrics[metric]} Test {metric_name}: {test_metrics[metric]} "
        data_writer.add_scalar(f"train/{metric_name}", train_metrics[metric], epoch)
        data_writer.add_scalar(f"val/{metric_name}", val_metrics[metric], epoch)
        data_writer.add_scalar(f"test/{metric_name}", test_metrics[metric], epoch)
    print(txt_log)