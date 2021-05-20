from paddle_fl.paddle_fl.core.trainer.fl_trainer import FLTrainerFactory
from paddle_fl.paddle_fl.core.master.fl_job import FLRunTimeJob
import paddle

import numpy
import sys
import logging
import time

from mlxtend.data import loadlocal_mnist
import json

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
trainer_id = int(sys.argv[1])  # trainer id
job_path = "fl_job_config"
job = FLRunTimeJob()
job.load_trainer_job(job_path, trainer_id)
job._scheduler_ep = "127.0.0.1:9091"  # Inform scheduler IP address to trainer
trainer = FLTrainerFactory().create_fl_trainer(job)
trainer._current_ep = "127.0.0.1:{}".format(9000 + trainer_id)
place = paddle.fluid.CPUPlace()
trainer.start(place)

test_program = trainer._main_program.clone(for_test=True)

#  Load data 
###############

def data_loader(data, idx):
    """
    It's a function that creates an iterable on the data of the client
    """
    def reader():
        for im in idx:
            yield data[0][im] / 255.0, int(data[1][im])
    return reader

with open('../data/MNIST/data_idx.json', 'r') as fp:      # Link to the file data_idx.json
    idx = json.load(fp)["client_{}".format(trainer_id)]

X_train, y_train = loadlocal_mnist(
            images_path='../data/MNIST/mnist_data/train-images-idx3-ubyte', 
            labels_path='../data/MNIST/mnist_data/train-labels-idx1-ubyte')     # Link to the folder contained data

all_samples_train = idx["train"]
train_reader = paddle.batch(reader=paddle.reader.shuffle(reader=data_loader([X_train, y_train], all_samples_train), buf_size=len(all_samples_train)), batch_size=64)

# Load test data if it's the client 0
if trainer_id == 0:
    X_test, y_test = loadlocal_mnist(
            images_path='../data/MNIST/mnist_data/t10k-images-idx3-ubyte', 
            labels_path='../data/MNIST/mnist_data/t10k-labels-idx1-ubyte')

    test_reader = paddle.batch(reader=data_loader([X_test, y_test], numpy.arange(len(y_test))), batch_size=64)

img = paddle.fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
label = paddle.fluid.layers.data(name='label', shape=[1], dtype='int64')
feeder = paddle.fluid.DataFeeder(feed_list=[img, label], place=paddle.fluid.CPUPlace())

# Accuracy function
#####################
def accuracy(train_test_program, train_test_feed, train_test_reader):
    acc_set = []
    avg_loss_set = []
    for test_data in train_test_reader():
        acc_np, avg_loss_np = trainer.exe.run(
            program=train_test_program,
            feed=train_test_feed.feed(test_data),
            fetch_list=["accuracy_0.tmp_0", "mean_0.tmp_0"])
        acc_set.append(float(acc_np))
        avg_loss_set.append(float(avg_loss_np))
    acc_val_mean = numpy.array(acc_set).mean()
    avg_loss_val_mean = numpy.array(avg_loss_set).mean()
    return avg_loss_val_mean, acc_val_mean


#  Parameters
################
num_epoch = 3
num_round = 200

#  Run
#########
round_id = 0
while not trainer.stop():
    round_id += 1

    if round_id > num_round:
        break

    trainer.run_with_epoch(reader = train_reader,feeder = feeder, fetch=["accuracy_0.tmp_0"], num_epoch = num_epoch)
    loss_train, acc_train = accuracy(test_program,feeder, train_reader)

    if trainer_id == 0:
        loss_val, acc_val = accuracy(test_program,feeder, test_reader)
        print("{} Round {} Train loss: {} Train accuracy: {} Test loss: {} Test accuracy: {}".format(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), round_id, loss_train, acc_train, loss_val, acc_val))
    else:
        print("{} Round {} Train loss: {} Train accuracy: {}".format(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), round_id, loss_train, acc_train))


    # Uncomment if you want to save the model of the clients
    # output_folder = "model_node%d" % trainer_id
    # save_dir = (output_folder + "/round_%d".format(round_id))
    # print("start save")
    # trainer.save_inference_program(output_folder)