import tensorflow as tf

class Multilayer_perceptron(object):
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
      Return a Multilayer perceptron model
      '''
      return tf.keras.models.Sequential([
            tf.keras.Input(shape=self.input_shape),
            tf.keras.layers.Dense(100, activation="relu"),
            tf.keras.layers.Dense(100, activation="relu"),
            tf.keras.layers.Dense(self.label_shape[-1], activation="softmax"),
      ])