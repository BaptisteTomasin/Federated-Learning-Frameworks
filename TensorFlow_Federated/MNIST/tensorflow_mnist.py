import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Delete warning if you don't have GPU

from mlxtend.data import loadlocal_mnist
import json

import keras_utils
import tensorflow as tf
import datetime
import models as md

# Args
#########

class Arguments():
    '''
    This class contain the different parameters of the FL strategy
    '''
    def __init__(self):
        self.batch_size = 64
        self.epochs =200
        self.lr = 0.1
        self.model = md.Multilayer_perceptron

args = Arguments()

# Load Data
##############

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

X_train, y_train = X_train[all_samples_train] / 255.0, y_train[all_samples_train]
X_test = X_test / 255.0

# Model
##########

model = args.model()

model.compile(
    optimizer=tf.keras.optimizers.SGD(args.lr),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

# Summary
############

log_dir = "tensorflow_logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train
##########

model.fit(
    X_train, y_train,
    epochs=args.epochs,
    validation_data=(X_test, y_test),
    callbacks=[tensorboard_callback],
    batch_size = args.batch_size
)

