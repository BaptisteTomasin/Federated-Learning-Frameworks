"""
    This file contains a template allowing to generate your personal model
"""
import paddle.fluid as fluid

class Dummy(object):
    def __init__(self):
        pass

    def lr_network(self, input_layer, label_layer):
        """
            Init the model
            arg:
                - input_layer: The input data layer
                - label_layer:  The label data layer
        """
        self.inputs = input_layer
        self.inputs += 0.0
        self.label = label_layer

        self.predict = self.dummy(self.inputs) 

        self.loss = None    # Your loss function
        self.acc = None     # Your accuracy function
        self.fetch_list = None  #Define the list of metrics name ex: [self.loss.name, self.accuracy.name]
        self.startup_program = fluid.default_startup_program()

    def dummy(self, inputs):        
        """
            Define the structure of your network
            arg: 
                - inputs: input layer
        """
        pass