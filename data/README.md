# Data

In this folder, we have the data use in the script of the different framework, particularly:
* The MNIST's dataset that contain images of digits
* The weather data in France, it corresponding at time-series data.

### MNIST

You can download the MNIST dataset with the following command:

    ./mnist.sh

This file downloads the dataset and stock it in a new folder named [mnist_data](/data/MNIST/mnist_data). To work with this dataset you will to install **mlxtend**, with the following command:

    pip install mlxtend

To do the experiment we have to distribute the data, with the file [data_idx_generator.py](/data/MNIST/data_idx_generator.py). You can run it with the following command:

    python data_idx_generator.py

This script returns the file **data_idx.json** that contains the index of MNIST images for all clients. The train data are chosen randomly in the MNIST dataset and for the test data we test our model on all the MNIST test data.

You can see below an extract of the file **data_idx.json**.

    {
        "client_0": {
            "train": [
                53037,
                42761,
                35136,
                ...
                38417,
                31678,
                3079
            ]
        },
        "client_1": {
            "train": [
                53037,
                42761,
                35136,
                ...
                38417,
                31678,
                3079
            ]
        },
        ...
    }

You can also activate the variable **save_hist** in [data_idx_generator.py](/data/MNIST/data_idx_generator.py), that allows you to display the histogram of the distribution and so check the non-IID of data between the different clients. For example, on three different clients, I obtain this distribution:

<p float="left", style="text-align: center;">
  <img src="/images/hist_client_2.png" width="300"/> 
  <img src="/images/hist_client_3.png" width="300"/> 
  <img src="/images/hist_client_6.png" width="300"/>
</p>

### Weather data