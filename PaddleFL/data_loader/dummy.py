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
        This function return an iterable on the train data 
        if client != None this use the train data associate at the client
        else it return an iterable on all the train data
        """
        pass

    def val_data(self, client = None):
        """
        This function return an iterable on the val data 
        if client != None this use the val data associate at the client
        else it return an iterable on all the val data
        """
        pass

    def test_data(self):
        """
        This function return an iterable on all the test data 
        """
        pass
