"""
    This file contains a template allowing to generate your personal data loader
"""
class Dummy:
    def __init__(self, data_path = None):
        """"
        data_path: path to the data
        You can add all args that you need, and you give their value in the config.json file
        """
        #Dataset info:
        self.data_path = data_path

        self.load()

    def load(self):
        """
            Allows to load the dataset
        """
        pass

    def train_data(self, client = None):
        """
        This function return tf.data.Dataset object if client != None this contains the train data associate at the client
        else it return the x_train array and the y_train array
        """
        pass

    def val_data(self, client = None):
        """
        This function return tf.data.Dataset object if client != None this contains the validation data associate at the client
        else it return the x_val array and the y_val array
        """
        pass

    def test_data(self):
        """
        This function return the x_test array and the y_test array
        """
        pass
