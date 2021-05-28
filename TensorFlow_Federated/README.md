# Tensorflow Federated of Google Inc

[Tensorflow Federated](https://github.com/tensorflow/federated) is an open source framework, developed by Google Inc. Based on the framework Tensorflow, it gives the possibility to simulate a Federated Learning. Tensorflow Federated allows to developed aggregates or learning methods. Tensoflow federated is composed by two layers:

* Federated Learning (FL) API, this layer is the high-level of the framework. It allows to do federated tranning and evaluation, with the utils give by the API 
* Federated Core (FC) API, this layer allows to test and creat new federated algorithm based on TensorFlow

Tensorflow Federated don't offer the privacy of data and the use of GPU. To add the privacy of data we can use [TensorFlowPrivacy](https://github.com/tensorflow/privacy) that allow to train machine-learning models with privacy 


In [MNIST](/TensorFlow_Federated/MNIST/) you can find the application of Tensorflow Federated on the MNIST dataset.


### Work environment

To beginning we will create our work environment. 
If you are on Ubuntu or MacOS, you can install it with this [instruction](https://www.tensorflow.org/federated/install)

Else you can also install [Docker](https://www.docker.com/), and after, to build the image with this Dockerfile the command is the following:

    docker build -t project/tff <directory to the Dockerfile>

This docker image allow you to run all TensorFlow Federated's script. For this, nothing could be easier, you run the docker image with the following command:

    docker run -it --rm -v <directory of the TensorFlow Federated's project>:/My_program/ -v <directory of the data folder>:/data/ project/tff /bin/sh

After it, your project is in the folder **My_program** of the docker and your data is in the folder **data** of the docker. To take more information about the use of Docker, you can see this [tutorial](https://docs.docker.com/get-started/).

### Analyse of this framework

Currently, TensorFlow Federated has big community with 240 issues and 73 contributor on GitHub. So, it's easy to find some helped or examples. The script is well commented that makes easy the comprehension of the different function or also the modification of the source script, like with the file [keras_utils.py](/TensorFlow_Federated/MNIST/keras_utils.py) where I modify the class named **_KerasModel's** to return the client metrics during the train or evaluation.

To evaluate this framework, I appliqued TensorFlow Federated on the MNIST dataset. You can see the commented script in the folder [MNIST](/TensorFlow_Federated/MNIST/). For my experience I use ten clients and I do three epochs by round.  The data of client is randomly selected on all the train dataset of MNIST, the distribution of data is done in the folder [data](/data). And to compare the accuracy of the model we test the model on the test dataset of MNIST.


Two models are proposed:
* A multilayer perceptron compose by two fully connected layer their size is 100, with the ReLU's activation function, and the fully connected output layer, the size is the label size (10) with the softmax's activation function.
* A convolutional neural network, compose by:
    * A first convolutional layer, the convolution kernel size is 3*3, a total of 32 convolution kernels
    * A first max pooling layer, the pooling size is 2*2, the step size is 1
    * A flatten layer
    * A first fully connected layer, with size 128
    * A fully connected output layer, the size is the label size (10) and with softmax as the activation function

All this models are defined in the file [models.py](/TensorFlow_Federated/MNIST/models.py)

In the following, a summary table of all my accuracies obtain on the test's dataset.

<table>
    <thead>
        <tr>
            <th colspan=3>Accuracy</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <th>Model</th>
            <th>TFF with Federated Averaging </th>
            <th>TensorFlow</th>
        </tr>
        <tr>
            <th>Multi-layer perceptron</th>
            <td>0.975 (~105 epochs)</td>
            <td>0.979 (~ 15 epochs)</td>
        </tr>
        <tr>
            <th> Convolutional NN </th>
            <td>0.980 (~ 100 rounds)</td>
            <td>0.984 (~15 epochs)</td>
        </tr>
    </tbody>
</table>

This table shows us that the Federated Averaging method don't have a big impact on the accuracy of the model. We lost approximately 0.04 in accuracy. But, the federated learning take more iterations, we can ask if this has a big impact on the time of convergences.

<table>
    <thead>
        <tr>
            <th colspan=3>Times</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <th>Model</th>
            <th>TFF with Federated Averaging</th>
            <th>Tensorflow</th>
        </tr>
        <tr>
            <th>Multi-layer perceptron</th>
            <td> 900s (00:15:00)</td>
            <td>66s (00:01:00)</td>
        </tr>
        <tr>
            <th> Convolutional NN </th>
            <td>7320s (02:20:00)</td>
            <td>480s (00:08:00)</td>
        </tr>
    </tbody>
</table>

This second table demonstrate that the run takes more time when we use the federated learning. Tensorflow federated don't have communication time because it offers just the possibility to made simulation of federated learning. So the cause of this slowdown is the number of loop more important, for converge. You can check the increase of times with the CNN (an important number of parameters) where the time for execution is fourteen times more important and the number of rounds is six times more important than the number of epochs.

We can say that this framework is a good tool to simulate some federated learning strategies because, with the different examples, issues and error report finding on the internet, it's easy to use it. The main problem of this framework that it doesn't offer the possibility to use it in deployment. But it's  updated quite often, we can hope to view quickly a deployment version.

The next step of the experiment can be to compare the performance of TensorFlow Federated with TensorFlow Privacy and show if it increases the time of execution and if it has a big impact on the accuracy of the model.