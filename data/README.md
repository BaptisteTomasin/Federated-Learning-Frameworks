# Data

This folder includes the data distribution used for my experiment:
* [mnist_clients_distribution_10_clients.json](/data/mnist_clients_distribution_10_clients.json) is an example of distribution on 10 clients file used for the MNIST_loader class. To build this the indices of train data are split on each client (25% of client indices are the validation) and the test data is used to test the model.
* [mnist_clients_distribution_50_clients.json](/data/mnist_clients_distribution_10_clients.json) is an example of distribution on 50 clients file used for the MNIST_loader class. To build this the indices of train data are split on each client (25% of client indices are the validation) and the test data is used to test the model.
* [Time_serie.csv](/data/Time_serie.csv) is the univariate ARMA series used for the time series experiment with Tensorflow Federated. [Best_predictor.csv](/data/Best_predictor.csv) is the best predictor associate to the ARMA series

### Notes

* To use the MNIST clients distribution refer the path of the json file in the config at the key name "distribution", else a new distribution will be generated.

* To use the Arma serie refer the path of the csv file in the config at the key name "data" and change the value of "distributed" key as false, after a new distribution will be generated.