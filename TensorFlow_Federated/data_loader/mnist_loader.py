from os.path import join 
import subprocess
import numpy
import matplotlib.pyplot as plt
from mlxtend.data import loadlocal_mnist
import json
import os
import tensorflow as tf
import collections

class MNIST_loader:
    def __init__(self,data = None, distribution = None, number_of_clients = None):
        """
            Load, Distribute and made the samples for time series data
            Arguments:
                data: [str] path to folder containing MNIST ('mnist_data/train-images-idx3-ubyte', 
                                                    'mnist_data/train-labels-idx1-ubyte',
                                                    'mnist_data/t10k-images-idx3-ubyte', 
                                                    'mnist_data/t10k-labels-idx1-ubyte')
                    if mnist_path == None: the data will be downloaded 

                distribution: [str] path to the json distribution file (clients_distribution.json)
                            if clients_path == None: a new distribution will be created

                number_of_clients: [int] number of clients for the distribution of data
        """
        #Dataset info:
        self.mnist_path = data

        #Distribution info
        self.clients_path = distribution
        self.number_of_clients = number_of_clients

        self.load_mnist()
        self.load_distribution()

    def load_mnist(self):
        """
            Load the data set
        """
        if self.mnist_path != None:
            self.__load_mnist()

        else:
            # Download MNIST dataset from https://data.deepai.org/mnist.zip 
            print("Download MNIST, this may take a little while, please be patient ")
            subprocess.call(['sh', './data_loader/mnist_downloader.sh'])
            self.mnist_path = "mnist_data"
            self.__load_mnist()

    def __load_mnist(self):
        """
            Load the mnist files
        """
        self._x_train, self._y_train = loadlocal_mnist(
                images_path=join(self.mnist_path,'train-images-idx3-ubyte'), 
                labels_path=join(self.mnist_path,'train-labels-idx1-ubyte'))

        self._x_test, self._y_test = loadlocal_mnist(
                images_path=join(self.mnist_path,'t10k-images-idx3-ubyte'), 
                labels_path=join(self.mnist_path,'t10k-labels-idx1-ubyte'))


    def load_distribution(self):
        """
            Load the distribution
        """
        if self.clients_path and self.number_of_clients == None:
            if os.path.exists(self.clients_path):
                with open(self.clients_path, 'r') as fp:
                    self.clients = json.load(fp)
                self.number_of_clients = len(self.clients.keys())
            else:
                print(f"Error: {self.clients_path} doesn't exist")
        else:
            if self.number_of_clients and self.clients_path and not os.path.exists(self.clients_path):
                self.__build_distribution(self.clients_path)
            elif self.number_of_clients and not self.clients_path:
                self.__build_distribution('clients_distribution.json')
            elif self.number_of_clients and self.clients_path and os.path.exists(self.clients_path):
                with open(self.clients_path, 'r') as fp:
                    self.clients = json.load(fp)
                if len(self.clients.keys()) != self.number_of_clients:
                    self.number_of_clients = len(self.clients.keys())
                    print("The number of clients is {self.number_of_clients}")
            else:
                print("Please, inform the clients_path or the number of clients")

    def __build_distribution(self, path, validation_rate = 0.25):
        """
            Distribute the MNIST samples on each client
            Arguments:
                path: [str] path to save the new distribution
                validation_rate: [float] percentage of clients data used as validation data
        """
        if hasattr(self, '_x_train') and hasattr(self, '_y_train'):
            num_imgs_per_client = len(self._y_train) // self.number_of_clients
            num_shards = len(self._y_train) / num_imgs_per_client * 100

            self.clients = {"client_{}".format(i): {"train": numpy.array([])} for i in range(self.number_of_clients)}
            idxs = numpy.arange(len(self._y_train))
            
            # sort labels
            idxs_labels = numpy.vstack((idxs, self._y_train))
            idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
            idxs = idxs_labels[0, :]

            idxs = numpy.array_split(idxs, num_shards)
            idx_shard = numpy.arange(num_shards)

            for i in range(self.number_of_clients):
                rand_set = set(numpy.random.choice(idx_shard, len(idxs)//self.number_of_clients, replace=False))
                idx_shard = list(set(idx_shard) - rand_set)
                data = numpy.array(idxs)[[int(k) for k in list(rand_set)]].reshape(-1).tolist()
                numpy.random.shuffle(data)
                self.clients[f"client_{i}"]["train"] = data[:-int(len(data) * validation_rate)]
                self.clients[f"client_{i}"]["val"] = data[-int(len(data) * validation_rate):]
            
            # Saving index dict
            with open(os.path.join(os.getcwd(), path), 'w') as fp:
                json.dump(self.clients, fp,  indent=4)

    def clients_distribution(self):
        """
            Retrun the distribution
        """
        return self.clients

    def clients_histogram(self, path):
        """
            Build the histogram of each client
            Arguments:
                path: [str] path to save the new distribution     
        """
        if not os.path.exists(path):
            os.makedirs(path)

        for client in self.clients:
            im_idx = self.clients[client]["train"] + self.clients[client]["val"]
            plt.figure()
            plt.hist(self._y_train[im_idx], bins = 10)
            plt.savefig(os.path.join(path, f"{client}.png"))
        
    def _data_loader(self,data, idx):
        '''
        This function builds the samples
        Arguments:
                data: [list] list of input and output data [x, y]
        '''
        data_x = []
        data_y = []
        for i in idx:
            x_floats = tf.constant(data[0][i] / 255.0, shape = (784), dtype = float)
            y_floats = tf.constant(data[1][i],shape = (1,), dtype = "int32")
            data_x.append(x_floats)
            data_y.append(y_floats)
        return data_x, data_y
        
    def train_data(self, client = None):
        '''
        This function return the train samples
        Arguments:
                client: [int] indice of the clients if its none all train data are returned
        '''
        if client != None:
            x,y = self._data_loader([self._x_train, self._y_train], self.clients[f"client_{client}"]["train"])
            return tf.data.Dataset.from_tensor_slices(collections.OrderedDict(x=x, y=y))
        else:
            _x = []
            _y = []
            for client in self.clients:
                x,y = self._data_loader([self._x_train, self._y_train], self.clients[client]["train"])
                _x += x
                _y += y
            return numpy.array(_x), numpy.array(_y)

    def val_data(self, client = None):
        '''
        This function return the validation samples
        Arguments:
                client: [int] indice of the clients if its none all validation data are returned
        '''
        if client != None:
            x,y = self._data_loader([self._x_train, self._y_train], self.clients[f"client_{client}"]["val"])
            return tf.data.Dataset.from_tensor_slices(collections.OrderedDict(x=x, y=y))
        else:
            _x = []
            _y = []
            for client in self.clients:
                x,y = self._data_loader([self._x_train, self._y_train], self.clients[client]["val"])
                _x += x
                _y += y
            return numpy.array(_x), numpy.array(_y)

    def test_data(self):
        '''
        This function return the test samples
        '''
        _idx  = numpy.arange(len(self._y_test))
        x, y = self._data_loader([self._x_test, self._y_test], _idx)
        return numpy.array(x), numpy.array(y)