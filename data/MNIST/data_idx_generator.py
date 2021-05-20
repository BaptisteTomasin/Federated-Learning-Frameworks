from mlxtend.data import loadlocal_mnist
import os
import matplotlib.pyplot as plt
import numpy
import json

#  Parameters
################

num_train_ex = None     # None if your use all data
num_clients = 10
save_hist = True

idx = {}

#  Loading data 
###############
X_train, y_train = loadlocal_mnist(
            images_path= os.path.join(os.getcwd(), 'mnist_data/train-images-idx3-ubyte'), 
            labels_path= os.path.join(os.getcwd(), 'mnist_data/train-labels-idx1-ubyte'))    # Link to the folder contained the data

#  Selecting Data
####################

def noniid_distri(y, num_clients, num_train_ex = None):
    """
    Return non-I.I.D client data from dataset, with the same number of examples per clients
    """

    num_imgs = num_train_ex if num_train_ex != None else len(y)//num_clients
    num_shards = len(y)/num_imgs*100

    dict_idx = {"client_{}".format(i): {"train": numpy.array([])} for i in range(num_clients)}
    idxs = numpy.arange(len(y))

    # sort labels
    idxs_labels = numpy.vstack((idxs, y))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    idxs = numpy.array_split(idxs, num_shards)
    idx_shard = numpy.arange(num_shards)

    for i in range(num_clients):
        rand_set = set(numpy.random.choice(idx_shard, len(idxs)//num_clients, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        dict_idx["client_{}".format(i)]["train"] = numpy.array(idxs)[[int(k) for k in list(rand_set)]].reshape(-1).tolist()
    return dict_idx

if __name__ == "__main__":

    idx = noniid_distri(y_train, num_clients)

    # Saving index dict
    with open(os.path.join(os.getcwd(), 'data_idx.json'), 'w') as fp:
        json.dump(idx, fp,  indent=4)

    if save_hist:
        if not os.path.exists(os.path.join(os.getcwd(), "hist")):
            os.makedirs(os.path.join(os.getcwd(), "hist"))

        for i in range(num_clients):
            plt.figure()
            plt.hist(y_train[idx["client_{}".format(i)]["train"]], bins = 10)
            plt.savefig(os.path.join(os.path.join(os.getcwd(), "hist"), "client_{}.png".format(i)))
