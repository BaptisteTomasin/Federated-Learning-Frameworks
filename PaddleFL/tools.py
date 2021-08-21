import numpy
import importlib
from inspect import signature


def metrics(exe, train_test_program, train_test_feed, train_test_reader, fetch_list):
    """
        Return the metric values on other data
        Arguments:
            exe : executor
            train_test_program: test program of the model
            train_test_feed: feed of the new data
            train_test_reader: reader of the new data
            fetch_list: fetch list of metrics
    """
    _metrics = []
    for test_data in train_test_reader():
        __metrics = exe.run(
            program=train_test_program,
            feed=train_test_feed.feed(test_data),
            fetch_list=fetch_list)
        _metrics.append(__metrics)
    _metrics_mean = numpy.array(_metrics).mean(axis = 0)
    return _metrics_mean.reshape(-1)

def select_model(model_name, inp, label, number_of_class = None):
    """
        Return the model chose in the configs
        Arguments:
            model_name: [str] name of the model class
            input_layer: [layer] input layer
            label_layer: [layer] output layer
    """
    model = getattr(importlib.import_module("models"), model_name)()
    args = signature(model.lr_network).parameters
    if "number_of_class" in args and number_of_class != None:
        model.lr_network(inp, label, number_of_class)
    else:
        model.lr_network(inp, label)
    return model

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
