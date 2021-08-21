import tensorflow as tf

class Simple_rnn(object):
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

        self.loss = tf.keras.losses.MeanSquaredError()
        self.metrics = [tf.keras.metrics.MeanSquaredError()]

    def build_model(self):
        '''
        Returns a Simple RNN model
        '''
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        Rnn1 = tf.keras.layers.SimpleRNN(10, return_state=True, activation = None)
        rnn_outputs1 = Rnn1(inputs)
        ts_inputs = tf.keras.layers.RepeatVector(self.label_shape[-2])(rnn_outputs1[0])
        outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.label_shape[-1]))(ts_inputs)
        model = tf.keras.models.Model(inputs,outputs)
        return model