import paddle.fluid as fluid

class Multilayer_perceptron(object):
    def __init__(self):
        pass

    def lr_network(self):
        self.inputs = fluid.layers.data(
            name='img', shape=[1, 28, 28], dtype="float32")
        self.label = fluid.layers.data(name='label', shape=[1], dtype='int64')

        self.predict = self.multilayer_perceptron(self.inputs) 
        self.sum_cost = fluid.layers.cross_entropy(
            input=self.predict, label=self.label)
        self.accuracy = fluid.layers.accuracy(
            input=self.predict, label=self.label)
        self.loss = fluid.layers.mean(self.sum_cost)
        self.startup_program = fluid.default_startup_program()

    # Define the structure of model
    def multilayer_perceptron(self, input):
        # The first fully connected layer, the activation function is ReLU
        hidden1 = fluid.layers.fc(input=input, size=100, act='relu')
        # The second fully connected layer, the activation function is ReLU
        hidden2 = fluid.layers.fc(input=hidden1, size=100, act='relu')
        # The fully connected output layer, the size is the label size (10) with the softmax's activation function.
        fc = fluid.layers.fc(input=hidden2, size=10, act='softmax')
        return fc


class Convolutional_NN(object):
    def __init__(self):
        pass

    def lr_network(self):
        self.inputs = fluid.layers.data(
            name='img', shape=[1, 28, 28], dtype="float32")
        self.label = fluid.layers.data(name='label', shape=[1], dtype='int64')

        self.predict = self.convolutional_neural_network(self.inputs) 
        self.sum_cost = fluid.layers.cross_entropy(
            input=self.predict, label=self.label)
        self.accuracy = fluid.layers.accuracy(
            input=self.predict, label=self.label)
        self.loss = fluid.layers.mean(self.sum_cost)
        self.startup_program = fluid.default_startup_program()

    # Define the structure of model
    def convolutional_neural_network(self, input):
        # The first convolutional layer, the convolution kernel size is 5*5, a total of 32 convolution kernels
        conv1 = fluid.layers.conv2d(input=input, num_filters=32, filter_size=3, act="relu")
        # The first pooling layer, the pooling size is 2*2, the step size is 1, and the maximum pooling
        pool1 = fluid.layers.pool2d(input=conv1, pool_size=2)
        # The flatten layer
        flatten = fluid.layers.flatten(pool1)
        # The first fully connected layer
        fc1 = fluid.layers.fc(input=flatten, size=128)
        # The fully connected output layer, the size is the label size (10) and with softmax as the activation function
        fc2 = fluid.layers.fc(input=fc1, size=10, act='softmax')
        return fc2