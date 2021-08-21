import tensorflow as tf
import numpy as np

class Convolutional_NN(object):
    def __init__(self):
        pass

    def lr_network(self, input_shape, label_shape):
        """
            Create loss function and the list of metrics
            Arguments:
                input_shape: [list / tuple] input shape
                label_shape: [list / tuple] output shape
        """

        self.label_shape = label_shape
        self.input_shape = input_shape

        self.loss = tf.keras.losses.SparseCategoricalCrossentropy()
        self.metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]

    def build_model(self):
        '''
        Return a CNN model
        '''
        model = tf.keras.models.Sequential()
        model.add( tf.keras.Input(shape=self.input_shape))
        if len(self.input_shape) == 1:
            model.add(tf.keras.layers.Reshape((int(np.sqrt(self.input_shape[-1])), int(np.sqrt(self.input_shape[-1])),1), input_shape=(784,)))
        model.add( tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128))
        model.add(tf.keras.layers.Dense(self.label_shape[-1], activation="softmax"))

        return model