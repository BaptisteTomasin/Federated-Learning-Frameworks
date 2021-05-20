import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Delete warning if you don't have GPU

from mlxtend.data import loadlocal_mnist
import json
import numpy as np
import collections

import keras_utils
import tensorflow as tf
import tensorflow_federated as tff
import models as md
import time

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Args
#########

class Arguments():
    '''
    This class contain the different parameters of the FL strategy
    '''
    def __init__(self):
        self.batch_size = 64
        self.rounds = 300
        self.epochs = 3
        self.shuffle_buffer = 100
        self.prefetch_buffer = 10
        self.lr = 0.1
        self.model = md.Convolutional_NN
args = Arguments()

# Load Data
##############

X_train, y_train = loadlocal_mnist(
            images_path='../data/MNIST/mnist_data/train-images-idx3-ubyte', 
            labels_path='../data/MNIST/mnist_data/train-labels-idx1-ubyte')     # Link to the folder contained data

X_test, y_test = loadlocal_mnist(
            images_path='../data/MNIST/mnist_data/t10k-images-idx3-ubyte', 
            labels_path='../data/MNIST/mnist_data/t10k-labels-idx1-ubyte')

X_test = X_test.reshape(-1, 28*28)
X_test = X_test.astype('float32') / 255.0

with open('../data/MNIST/data_idx.json', 'r') as fp:      # Link to the file data_idx.json
    idx = json.load(fp)    

def preprocess_mnist(X, Y, idx, args):
    '''
    This function builds the data-set of the client
    Arg:
      X: Global input
      Y: Global output
      idx: Idx of input and output associate to the client
      args: Parameters of the FL strategy
    '''
    x = []
    y = []
    for i in range(0, len(idx), args.batch_size):
        batch_samples = idx[i:i + args.batch_size]
        x += [tf.constant(X[j] / 255.0, shape = (784), dtype = float) for j in batch_samples]
        y += [tf.constant(Y[j],shape = (1,), dtype = "int32") for j in batch_samples]
    return tf.data.Dataset.from_tensor_slices(collections.OrderedDict(x=x, y=y)).repeat(args.epochs).shuffle(args.shuffle_buffer).batch(
      args.batch_size).prefetch(args.prefetch_buffer)

federated_train_data = [preprocess_mnist(X_train, y_train, idx["client_{}".format(i)]["train"], args) for i in range(len(idx.keys()))]

# Model
##########

def create_tff_model():
  '''
  This function transforms the TF's model into the TFF's model
  '''
  keras_model = args.model()
  return keras_utils.from_keras_model(
      keras_model,
      input_spec=federated_train_data[0].element_spec,
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

# Federated Algorithm
########################
federated_algorithm = tff.learning.build_federated_averaging_process(
    model_fn=create_tff_model,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(lr=args.lr))


# Take metrics 
################

def print_metrics(round, metrics):
  txt = 'round {}, '.format(round)
  for name, value in metrics.items():
    if name[0:4] != 'per_':
      if name == 'num_examples':
        txt += name +' = {:.0f} '.format(value/args.epochs)
      else:
        txt += name +' = {:.4f} '.format(value)
  print(txt)

# Summary
############

# Log Folder 
logdir = "federated_logs/"
Summary_writer_train = [tf.summary.create_file_writer(logdir + "/train/serveur")]

# Création des Summary
for client in range(len(idx.keys())):
    Summary_writer_train += [tf.summary.create_file_writer(logdir + "/train/client_"+str(client))]

def makelog(metrics, round_num, Summary_writer, benchmark = 0.99):
    '''
    This function builds the log of the different clients and the server 
    Args:

      metrics: The metrics' output of the traning
      round_num: The round id
      Summary_writer: The list of summary associate to the clients and server.
      benchmark: this allows to display the percentage of clients where the metrics is above this benchmark
    '''

    for name, value in metrics.items():
        count = 1
        if name[0:4] != 'per_':
          if name == "num_examples":
            with Summary_writer[0].as_default():
              tf.summary.scalar(name, value/args.epochs, step=round_num)
          else:
            with Summary_writer[0].as_default():
                tf.summary.scalar(name, value, step=round_num)
        else:
          stats = 0
          for i in value:
            if i>= benchmark:
              stats += 1 
            if name[11:] == "num_examples":
                with Summary_writer[count].as_default():
                  tf.summary.scalar(name[11:], i/args.epochs, step=round_num)
            else:                                
              with Summary_writer[count].as_default():
                  tf.summary.scalar(name[11:], i, step=round_num)
            count+=1
          if name[11:] != "num_examples" and name[11:] != "loss":
            with Summary_writer[0].as_default():
              tf.summary.scalar(f"% of client > {benchmark} for {name[11:]}", stats/(len(Summary_writer)-1), step=round_num)

    return Summary_writer

# Run
########

server_state = federated_algorithm.initialize()

for round_num in range(1, args.rounds+1):
    print("{} :".format(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))))
    server_state, metric = federated_algorithm.next(server_state, federated_train_data)
    print_metrics(round_num, metric['train'])

    Summary_writer_train = makelog(metric['train'], round_num, Summary_writer_train)


    test_model = args.model()
    server_state.model.assign_weights_to(test_model)
    y_pred = test_model.predict(X_test.reshape(-1,784))
    loss_test = tf.keras.losses.sparse_categorical_crossentropy(y_test, y_pred.reshape(-1,10))
    acc_test = tf.keras.metrics.SparseCategoricalAccuracy() 
    acc_test.update_state(y_test.reshape(-1,1), y_pred.reshape(-1,10))
    print("Test loss: {}, Test Accuracy: {}".format(np.mean(loss_test.numpy()), acc_test.result().numpy()))

    with Summary_writer_train[0].as_default():
      tf.summary.scalar("Test loss", np.mean(loss_test.numpy()), step=round_num)
      tf.summary.scalar("Test Accuracy", acc_test.result().numpy(), step=round_num)