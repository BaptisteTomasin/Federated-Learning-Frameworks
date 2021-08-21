import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Delete warning if you don't have GPU

import json
import numpy
import tensorflow as tf
import argparse
from tools import select_model, select_data


parser = argparse.ArgumentParser()
parser.add_argument('--config_path', help = "path to the config file")
args = parser.parse_args()

with open(args.config_path, 'r') as fp: 
    params = json.load(fp)

# Load Data
##############
dataset = select_data(params)

x_train, y_train = dataset.train_data()
x_val, y_val = dataset.val_data()
x_test, y_test = dataset.test_data()

#  Define Model
#################
_Model = select_model(params["centralized"]["model_name"])
_Model.lr_network(params["centralized"]["input_shape"], params["centralized"]["label_shape"])
model = _Model.build_model()
model.compile(
    optimizer = tf.keras.optimizers.SGD(params["centralized"]["learning_rate"]),
    loss = _Model.loss,
    metrics = _Model.metrics,
)


# Create callback for test data
class Test_callback(tf.keras.callbacks.Callback):
    def __init__(self,log_dir, test_data):
        self._x_test, self._y_test= test_data
        self._summary = tf.summary.create_file_writer(log_dir + "/test")
        
    def on_epoch_end(self, epoch, logs = {}):
        y_pred = self.model.predict(self._x_test).reshape(params["centralized"]["label_shape"])
        loss_test = _Model.loss(self._y_test, y_pred)
        with self._summary.as_default():
            tf.summary.scalar("Test loss", numpy.mean(loss_test.numpy()), step = epoch)
        txt_test_log = "Test loss:  {:.4f} ".format(numpy.mean(loss_test.numpy()))
        for metric in _Model.metrics:
            metric_value = metric(self._y_test, y_pred).numpy()
            with self._summary.as_default():
                tf.summary.scalar(f"Test {metric.name}", metric_value, step = epoch)
            txt_test_log += "Test {}:  {:.4f} ".format(metric.name, metric_value)
        print(txt_test_log)

# Summary
############
log_dir = params["centralized"]["logdir"]
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = log_dir, histogram_freq = 1)
test_callback = Test_callback(log_dir = log_dir, test_data =  [x_test, y_test])

# Train
##########
history = model.fit(
    x_train, y_train, 
    epochs = params["centralized"]["num_epoch"], 
    batch_size = params["centralized"]["batch_size"], 
    validation_data = (x_val, y_val), 
    verbose = 1, 
    callbacks = [tensorboard_callback, test_callback],
    shuffle = False
)

