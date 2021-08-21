import tensorflow as tf
import time
import importlib
from inspect import signature

def print_metrics(round, metrics):
  '''
    Print the centralized metrics 
    Arguments:
      metrics: The metrics' output of the traning
      round_num: The round id
      summaries_writer: The list of summary associate to the clients and server.
  '''
  txt_log = "{} Round {} ".format(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),
                                     round)
  for name, value in metrics.items():
    if 'per_client' not in name:
      if 'num_examples' not in name:
        txt_log += "{} = {:.4f} ".format(name, value)
  print(txt_log)

def make_log(metrics, round_num, summaries_writer):
    '''
    Build the log of the different clients and the server 
    Arguments:
      metrics: The metrics' output of the traning
      round_num: The round id
      summaries_writer: The list of summary associate to the clients and server.
    '''
    for name, metric in metrics.items():
        count = 1
        if 'per_client' not in name:
          if "num_examples" not in name:
            with summaries_writer[0].as_default():
                tf.summary.scalar(name, metric, step=round_num)
        else:
            for count, value in enumerate(metric):
                if "num_examples" not in name:                               
                    with summaries_writer[count].as_default():
                        tf.summary.scalar(name[11:], value, step=round_num)
    return summaries_writer


def select_model(model_name):
    """
        Return the model chose in the configs
        Arguments:
            model_name: [str] name of the model class
    """
    return getattr(importlib.import_module("models"), model_name)()

def select_data(config):
    """
        Return the model chose in the configs
        Arguments:
            config: [dict] config dictionary
    """
    strategy = config["strategy"]
    _class = getattr(importlib.import_module("data_loader"), config[strategy]["data_loader"])
    args = signature(_class).parameters
    input_args = []
    check_args = []

    for arg in args:
        if arg not in config[strategy]:
            check_args += [arg]
        else:
            input_args += [config[strategy][arg]]

    if len(check_args) == 0:
        return _class(*input_args)
    else:
        print(f"Please add the keys {check_args} in config.json")
        exit()
