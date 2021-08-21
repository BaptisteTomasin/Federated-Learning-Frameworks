import paddle.fluid as fluid


class Multilayer_perceptron(object):
    def __init__(self):
        pass

    def lr_network(self, input_layer, label_layer, number_of_class = None):
        """
            Create loss function, the list of metrics and the fetch_list
            Arguments:
                input_layer: [layer] input layer
                label_layer: [layer] output layer
                number_of_class: [int] number of class predict
        """
        self.label_shape = label_layer.shape
        self.input_shape = input_layer.shape
        self.inputs = input_layer
        self.label = label_layer
        if number_of_class:
            self.number_of_class = number_of_class

        self.predict = self.multilayer_perceptron(self.inputs) 

        self.sum_cost = fluid.layers.cross_entropy(
            input = self.predict, label = self.label)

        self.accuracy = fluid.layers.accuracy(
            input = self.predict, label = self.label)
        self.loss = fluid.layers.mean(self.sum_cost)
        self.fetch_list = [self.loss.name, self.accuracy.name]
        self.startup_program = fluid.default_startup_program()

    def multilayer_perceptron(self, input):
        """
            Define the MP structure
            Arguments:
                input: [layer] input layer
        """
        # The first fully connected layer, the activation function is ReLU
        hidden1 = fluid.layers.fc(input = input, size = 100, act ='relu')
        # The second fully connected layer, the activation function is ReLU
        hidden2 = fluid.layers.fc(input = hidden1, size = 100, act ='relu')
        # The fully connected output layer with the softmax's activation function.
        if self.number_of_class:
            fc = fluid.layers.fc(input = hidden2, size = self.number_of_class, act ='softmax')
        else:
            fc = fluid.layers.fc(input = hidden2, size = self.label_shape[-1], act ='softmax')
        return fc