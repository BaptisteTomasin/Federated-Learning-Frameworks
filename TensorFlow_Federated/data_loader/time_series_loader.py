import numpy
import pandas as pd
import collections
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

class Time_series_loader:
    def __init__(self, data = None, distributed = False, number_of_clients = None, lookback = 30, lookforward = 1):
        """
            Load, Normalize, Distribute and made the samples for time series data
            Arguments:
                data: [str] path to data as csv type
                distributed: [bool] specify if data are already distributed
                number_of_clients: [int] number of clients for the distribution of data
                lookback: [int]
                lookforward: [int]
        """

        #Dataset info:
        self.ts_path = data
        self.distributed = distributed
        self.lookback = lookback
        self.lookforward = lookforward
        self.scalers={}

        #Distribution info
        self.number_of_clients = number_of_clients

        self.load_ts()
        if not self.distributed:
            self.__build_distribution()
        self._normalize()
        
    def load_ts(self):
        """
            Load the data set
        """
        if self.ts_path and self.distributed:
            self.clients = pd.read_csv(self.ts_path, index_col = 0, header = [0,1] )
        elif self.ts_path and not self.distributed:
            self.clients = pd.read_csv(self.ts_path, index_col = 0, delimiter= ",")
        else:
            print("Please give the csv contain the time series")

    def __build_distribution(self, path = "clients_distribution.csv", test_rate = 0.30, validation_rate = 0.25):
        """
            Distribute the Time Series on each client
            Arguments:
                path: [str] path to save the new distribution
                test_rate: [float] percentage of data used as test data
                validation_rate: [float] percentage of clients data used as validation data

        """
        ts_number = len(self.clients.columns)
        ts_size_min = min([len(self.clients[column]) for column in self.clients.columns])
        ts_size_per_client = (ts_size_min - int(ts_size_min*test_rate)) // self.number_of_clients
        validation_size = int(ts_size_per_client * validation_rate)

        # Buid the header name
        header1 = []
        for i in range (self.number_of_clients):
            header1 += [f"client_{i}"] * (ts_number) + [f"client_{i}_val"] * (ts_number)
        header1 += ["Test"] * (ts_number)
        dataframe_header = [header1, 
                            numpy.array([[ts_name] for ts_name in self.clients.columns] * (self.number_of_clients * 2 + 1)).reshape(-1)] 
        
        # Dispatch data on each client
        _data = pd.DataFrame(columns = dataframe_header)
        for ts_name in self.clients.columns:
            _data["Test", ts_name] = self.clients[ts_name].values[ts_size_min - int(ts_size_min * test_rate):ts_size_min]
            for c in range(self.number_of_clients):
                _data[f"client_{c}",ts_name] = self.clients[ts_name].iloc[c * ts_size_per_client : (c + 1) * ts_size_per_client - validation_size].reset_index(drop = True)
                _data[f"client_{c}_val",ts_name] = self.clients[ts_name].iloc[(c + 1) * ts_size_per_client - validation_size : (c + 1) * ts_size_per_client].reset_index(drop = True)
        
        self.clients = _data

        #Save the distribution
        self.clients.to_csv(path)

    def _normalize(self):
        """
            Normalize the distributed Time Series with the MinMaxScaler method
        """
        for ts in self.clients.columns.droplevel(0).unique():
            data = self.clients.loc(axis=1)[self.clients.columns.droplevel(1)!="Test",ts].values.reshape(-1)
            data = data[~numpy.isnan(data)]
            scaler = MinMaxScaler(feature_range=(-1,1))
            scaler.fit(data.reshape(-1,1))
            self.scalers['scaler_'+ ts] = scaler
        for client in self.clients.columns.droplevel(1).unique():
            for ts in self.clients.columns.droplevel(0).unique():
                scaler = self.scalers['scaler_' + ts]
                s_s = scaler.transform(self.clients[client, ts].values.reshape(-1,1))
                s_s = numpy.reshape(s_s, len(s_s))
                self.clients[client, ts] = s_s

    def _shift_samples(self, data):
        '''
        This function builds the samples
        Arguments:
                data: [list] list of input and output data [x, y]
        '''
        data = data.values
        X, y = list(), list()
        for window_start in range(len(data)):
            past_end = window_start + self.lookback
            future_end = past_end + self.lookforward
            if future_end > len(data):
                break
            # slicing the past and future parts of the window
            past, future = data[window_start:past_end, :], data[past_end:future_end, :]
            X.append(past)
            y.append(future)
        return X, y

    def train_data(self, client = None):
        '''
        This function return the train samples
        Arguments:
                client: [int] indice of the clients if its none all train data are returned
        '''
        if client != None:
            x,y = self._shift_samples(self.clients[f"client_{client}"].dropna())
            return tf.data.Dataset.from_tensor_slices(collections.OrderedDict(x=x, y=y))
        else:
            _x = []
            _y = []
            for column in self.clients.columns.droplevel(1).unique():
                if("val" not in column and "Test" not in column):
                    x,y = self._shift_samples(self.clients[column].dropna())
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
            x,y = self._shift_samples(self.clients[f"client_{client}_val"].dropna())
            return tf.data.Dataset.from_tensor_slices(collections.OrderedDict(x=x, y=y))
        else:
            _x = []
            _y = []
            for column in self.clients.columns.droplevel(1).unique():
                if("val" in column and "Test" not in column):
                    x,y = self._shift_samples(self.clients[column].dropna())
                    _x += x
                    _y += y
            return numpy.array(_x), numpy.array(_y)

    def test_data(self):
        '''
        This function return the test samples
        '''
        x,y = self._shift_samples(self.clients[f"Test"].dropna())
        return numpy.array(x), numpy.array(y)
