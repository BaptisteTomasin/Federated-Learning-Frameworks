# Tensorflow Federated of Google Inc

[Tensorflow Federated](https://github.com/tensorflow/federated) is an open-source framework, developed by Google Inc. Based on Tensorflow, it allows  to simulate a Federated Learning. Tensorflow Federated allows to developed aggregates or learning methods. Tensoflow federated is composed by two layers:

* Federated Learning (FL) API, this layer is the high-level of the framework. It allows doing federated training and evaluation, with the utils give by the API 
* Federated Core (FC) API, this layer allows to test and devise new federated algorithms based on TensorFlow

Tensorflow Federated don't offer the privacy of data and the use of GPU. To add the privacy of data, we can add [TensorFlowPrivacy](https://github.com/tensorflow/privacy) that allow to train machine-learning models with privacy.


In [MNIST](/TensorFlow_Federated/MNIST/) you can find the application of Tensorflow Federated on the MNIST dataset.


### Work environment

To beginning, we will establish our work environment. 
With Ubuntu or MacOS, you can install it with this [instruction](https://www.tensorflow.org/federated/install)

Else you can also install [Docker](https://www.docker.com/), and build the image, from the Dockerfile, with the following command:

    docker build -t project/tff <directory to the Dockerfile>

This docker image allows you to execute all TensorFlow Federated script. For this, nothing could be easier, you run the docker image with the following command:

    docker run -it --rm -v <directory of the TensorFlow Federated project>:/My_program/ -v <directory of the data folder>:/data/ project/tff /bin/bash

After in the docker, your project is in the folder **My_program** and your data in the folder **data**. To acquire more information about the use of Docker, you can observe this [tutorial](https://docs.docker.com/get-started/).

### Analyze of this framework

Currently, TensorFlow Federated has massive community with 240 issues and 73 contributor on GitHub. Consequently, it's effortless to find some helped or examples. The script is well commented that makes easy the comprehension of the diverse function or also the modification of the source script, like with the file [keras_utils.py](/TensorFlow_Federated/MNIST/keras_utils.py) where I modify the class named **_KerasModel's** to return the client metrics during the train or evaluation.

To evaluate this framework, I appliqued TensorFlow Federated on the MNIST dataset. You can follow the commented script in the folder [MNIST](/TensorFlow_Federated/MNIST/). For my experience I manage ten clients and do three epochs by round.  The data of the client are randomly selected on all the train dataset of the MNIST, the distribution of the data is done in the folder [data](/data). And to compare the accuracy of the model we test the model on the test dataset of MNIST.


Two models are proposed:
* A multilayer perceptron compose by two fully connected layer their size is 100, with the ReLU's activation function, and the fully connected output layer, the size is the label size (10) with the softmax's activation function.
* A convolutional neural network, compose by:
    * A first convolutional layer, the convolution kernel size is 3*3, a total of 32 convolution kernels
    * A first max pooling layer, the pooling size is 2*2, the step size is 1
    * A flatten layer
    * A first fully connected layer, with size 128
    * A fully connected output layer, the size is the label size (10) and with softmax as the activation function

All these models are defined in the file [models.py](/TensorFlow_Federated/MNIST/models.py)

In the following, a summary table of all my accuracies gets on the test's dataset.

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

This table demonstrates that the Federated Averaging method doesn't provide an enormous impact on the accuracy of the model. We lost approximately 0.04 in accuracy. But, the federated learning take more iterations, we can ask if this produces a significant impact on the time of convergences.

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

This second table demonstrates the run requires more time when we use the federated learning. Tensorflow federated don't have communication time because it offers just the possibility to made simulation of federated learning. Therefore, the cause of this slowdown is the number of loops more important, for converging. You can check the increase of times with the CNN (an important number of parameters). In fact, the time for execution is fourteen times more important and the number of rounds is six times more important than the number of epochs.

We can say this framework remains a good tool to simulate some federated learning strategies because with the different examples, issues and error report finding on the internet, it's easy to use it. The principal problem of this framework that it doesn't offer deployment mode. But frequently new updates are published, we can hope to view quickly a deployment version.

The next step of the experiment can be to compare the performance of TensorFlow Federated with TensorFlow Privacy and show if it increases the time of execution and if it includes an important impact on the accuracy of the model.