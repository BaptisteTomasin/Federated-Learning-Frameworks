import os
import numpy as np
import json
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Delete warning if you don't have GPU

from federated.my_keras_model import My_keras_model
from tools import select_model, select_data, print_metrics, make_log

import tensorflow as tf
import tensorflow_federated as tff

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', help = "path to the config file")
args = parser.parse_args()

with open(args.config_path, 'r') as fp: 
    params = json.load(fp)

# Load Data
##############
dataset = select_data(params)

federated_val_data = []
federated_train_data = []

for client in range(params["federated"]["number_of_clients"]):
  federated_val_data += [dataset.train_data(client = client).repeat(params["federated"]["num_epoch"]).batch(
      params["federated"]["batch_size"])]
  federated_train_data += [dataset.val_data(client = client).batch(
      params["federated"]["batch_size"])]

x_test, y_test = dataset.test_data()

# Model
##########
def create_tff_model():
  '''
  This function transforms the TF's model into the TFF's model
  '''
  keras_model = select_model(params["federated"]["model_name"])
  keras_model.lr_network(params["federated"]["input_shape"], params["federated"]["label_shape"])

  return My_keras_model(
          keras_model.build_model(),
          input_spec = federated_train_data[0].element_spec,
          loss_fns = [keras_model.loss],
          loss_weights = [1.0],
          Metrics = keras_model.metrics)

# Federated Algorithm
########################
federated_algorithm = tff.learning.build_federated_averaging_process(
    model_fn = create_tff_model,
    client_optimizer_fn = lambda: tf.keras.optimizers.SGD(lr = params["federated"]["learning_rate"]))

# Summary
############

logdir = params["federated"]["logdir"]
summaries_writer_train = [tf.summary.create_file_writer(logdir + "/train/serveur")]
summaries_writer_val = [tf.summary.create_file_writer(logdir + "/validation/serveur")]
summary_writer_test = tf.summary.create_file_writer(logdir + "/test/serveur")

# Création des Summary
for client in range(params["federated"]["number_of_clients"]):
    summaries_writer_train += [tf.summary.create_file_writer(logdir + "/train/client_"+str(client))]
    summaries_writer_val += [tf.summary.create_file_writer(logdir + "/validation/client_"+str(client))]

# Run
########
server_state = federated_algorithm.initialize()
validation = tff.learning.build_federated_evaluation(create_tff_model)

for round_num in range(params["federated"]["num_round"]):

    server_state, metric = federated_algorithm.next(server_state, federated_train_data)    
    val_metrics = validation(server_state.model, federated_val_data)
    print_metrics(round_num, metric['train'])

    # Add metrics on summaries
    summaries_writer_train = make_log(metric['train'], round_num, summaries_writer_train)
    summaries_writer_val = make_log(val_metrics, round_num, summaries_writer_val)

    # Test model on data test
    Test_model = select_model(params["federated"]["model_name"])
    Test_model.lr_network(params["federated"]["input_shape"], params["federated"]["label_shape"])
    test_model = Test_model.build_model()
    server_state.model.assign_weights_to(test_model)

    y_pred = test_model.predict(x_test).reshape(params["federated"]["label_shape"])
    loss_test = Test_model.loss(y_test, y_pred)
    with summary_writer_test.as_default():
        tf.summary.scalar("Test loss", np.mean(loss_test.numpy()), step = round_num)
    txt_test_log = "Test loss:  {:.4f} ".format(np.mean(loss_test.numpy()))
    for metric in Test_model.metrics:
        metric_value = metric(y_test, y_pred).numpy()
        with summary_writer_test.as_default():
            tf.summary.scalar(f"Test {metric.name}", metric_value, step = round_num)
        txt_test_log += "Test {}:  {:.4f} ".format(metric.name, metric_value)
    print(txt_test_log)