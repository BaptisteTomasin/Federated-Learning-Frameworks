import paddle.fluid as fluid

class Convolutional_NN(object):
    def __init__(self):
        pass

    def lr_network(self, input_layer, label_layer, number_of_class = None):
        """
            Create loss function, the list of metrics and the fetch_list
            Arguments:
                input_layer: [layer] input layer
                label_layer: [layer] output layer
        """
        self.label_shape = label_layer.shape
        self.input_shape = input_layer.shape
        self.inputs = input_layer
        self.label = label_layer
        if number_of_class:
            self.number_of_class = number_of_class

        self.predict = self.convolutional_neural_network(self.inputs) 

        self.sum_cost = fluid.layers.cross_entropy(
            input = self.predict, label = self.label)

        self.accuracy = fluid.layers.accuracy(
            input = self.predict, label = self.label)
        self.loss = fluid.layers.mean(self.sum_cost)
        self.fetch_list = [self.loss.name, self.accuracy.name]
        self.startup_program = fluid.default_startup_program()

    def convolutional_neural_network(self, input):
        """
            Define the CNN structure
            Arguments:
                input: [layer] input layer
        """
        # The first convolutional layer, the convolution kernel size is 5*5, a total of 32 convolution kernels
        conv1 = fluid.layers.conv2d(input=input, num_filters=32, filter_size=3, act="relu")
        # The first pooling layer, the pooling size is 2*2, the step size is 1, and the maximum pooling
        pool1 = fluid.layers.pool2d(input=conv1, pool_size=2)
        # The flatten layer
        flatten = fluid.layers.flatten(pool1)
        # The first fully connected layer
        fc1 = fluid.layers.fc(input=flatten, size=128)
        # The fully connected output layer, the size is the label size (10) and with softmax as the activation function
        if self.number_of_class:
            fc2 = fluid.layers.fc(input = fc1, size = self.number_of_class, act ='softmax')
        else:
            fc2 = fluid.layers.fc(input = fc1, size = self.label_shape[-1], act ='softmax')
        return fc2