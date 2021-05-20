import tensorflow as tf

def Multilayer_perceptron():
  '''
  This function returns a Multilayer perceptron model
  '''
  return tf.keras.models.Sequential([
        tf.keras.Input(shape=[784]),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
  ])

def Convolutional_NN():
  '''
  This function returns a CNN model
  '''
  return tf.keras.models.Sequential([
       tf.keras.Input(shape=[784]),
        tf.keras.layers.Reshape((28, 28,1), input_shape=(784,)),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128),
        tf.keras.layers.Dense(10, activation="softmax"),
  ])