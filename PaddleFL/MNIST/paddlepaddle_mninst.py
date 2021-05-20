import paddle as paddle
from models import *

from mlxtend.data import loadlocal_mnist
import json
import numpy
import time

#  Define Model
#################

model = Multilayer_perceptron()
model.lr_network()

#  Load data 
###############
X_train, y_train = loadlocal_mnist(
            images_path='../data/MNIST/mnist_data/train-images-idx3-ubyte', 
            labels_path='../data/MNIST/mnist_data/train-labels-idx1-ubyte')     # Link to the folder contained data

X_test, y_test = loadlocal_mnist(
            images_path='../data/MNIST/mnist_data/t10k-images-idx3-ubyte', 
            labels_path='../data/MNIST/mnist_data/t10k-labels-idx1-ubyte')

with open('../data/MNIST/data_idx.json', 'r') as fp:      # Link to the file data_idx.json
    idx = json.load(fp)

all_samples_train = []

for client in idx.keys():
    all_samples_train += idx[client]["train"]

def data_loader(data, idx):
    """
    It's a function that creates an iterable on the data of the client
    """
    def reader():
        for im in idx:
            yield data[0][im] / 255.0, int(data[1][im])
    return reader

train_reader = paddle.batch(reader=paddle.reader.shuffle(reader=data_loader([X_train, y_train], all_samples_train), buf_size=len(all_samples_train)), batch_size=64)
test_reader = paddle.batch(reader=data_loader([X_test, y_test], numpy.arange(len(y_test))), batch_size=64)

img = paddle.fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
label = paddle.fluid.layers.data(name='label', shape=[1], dtype='int64')
feeder = paddle.fluid.DataFeeder(feed_list=[img, label], place=paddle.fluid.CPUPlace())


#   Train
###############
test_program = paddle.fluid.default_main_program().clone(for_test=True)

# Define optimization method
optimizer = paddle.fluid.optimizer.SGD(learning_rate=0.1)
opts = optimizer.minimize(model.loss)

# Define a parser that uses the CPU
place = paddle.fluid.CPUPlace()
exe = paddle.fluid.Executor(place)
#Parameter initialization
exe.run(paddle.fluid.default_startup_program())

# Accuracy function
#####################
def accuracy(train_test_program, train_test_feed, train_test_reader):
    acc_set = []
    avg_loss_set = []
    for test_data in train_test_reader():
        acc_np, avg_loss_np = exe.run(
            program=train_test_program,
            feed=train_test_feed.feed(test_data),
            fetch_list=["accuracy_0.tmp_0", "mean_0.tmp_0"])
        acc_set.append(float(acc_np))
        avg_loss_set.append(float(avg_loss_np))
    acc_val_mean = numpy.array(acc_set).mean()
    avg_loss_val_mean = numpy.array(avg_loss_set).mean()
    return avg_loss_val_mean, acc_val_mean

# Epochs 
num_epoch = 100
#Start training and testing
for epoch in range(num_epoch):

    for data in train_reader():
        exe.run(program=fluid.default_main_program(),
                                        feed=feeder.feed(data),
                                        fetch_list=[model.loss, model.accuracy])

    loss_train, acc_train = accuracy(test_program,feeder, train_reader)

    loss_val, acc_val = accuracy(test_program,feeder, test_reader)

    print("{} Epoch {} Train loss: {} Train accuracy: {} Test loss: {} Test accuracy: {}".format(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), epoch, loss_train, acc_train, loss_val, acc_val))
