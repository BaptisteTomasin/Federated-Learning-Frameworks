# Federated Learning (FL) Frameworks

The Machine Learning is the fact to offer at the computer  the capacity to "learn" from data. We can separate it in two party. In a first time we build the model from learning data, that we know and that we can analyze it to choose the better model for our task. The second task is the deployment of the model, when our model is optimize and gives us the best score on the validation data, we use it on production.

Nowadays, for the learning data is collected and centralize by companies, but this approached is not always applicable because in some sectors data are protected for example in Banks or in Medical. to solve this problem a new approach of learning are in development, the Federated Learning.

In this Github we find applications of different FL frameworks like Tensorflow Federated, Pysift and PaddleFL.
Before beginning to discuss these frameworks, I want to present quickly the idea of the Federated Learning.

### Federated learning main idea and documents

The approach of Federated Learning is to send the model to the device instead of sending the data of device to the server. The traditional approach is composed by one server and multiple client (= worker). This learning method is an iterative method where each iteration, named a round or a cycle, is composed by four step:

* Step 1: The model is sent to the clients eligible for the round.

* Step 2: Each client trained locally the model with their data.

* Step 3: The model is sand back to the server, using encrypted communication that assure the privacy of data.

* Step 4: The server aggregat the model to form a new global model with all model trained locally.

After this learning step, like with a centralize learning, the server sends the global model to the clients that use it, it's the deployment task.

The derivative of this approach is without the server, the model is sent between the different clients with peer-to-peer connections. 

Some benefits are offered by the federated learning. The principal is that personal data remain local so, with the encoding of the local model before the send, data remains private. Moreover, the model is sent to the client to train it, so it's possible to offer a real-time prediction. A third benefits with the federated learning is the reduce hardware infrastructure required for the formation. Indeed, on the fact that the model is learned on the clients' device, is not necessary to have big computational resources to build a machine learning model.

But, federated learning have also some challenges to be solved. The principal challenges are :
* **The communication between sever and clients**, the technics to reduce the times of communication touch many parameters like the number of rounds, the number of epochs the complexity of the model ...
* **The variability of the clients**, all clients are not always open to participate at the round or also a client can be disconnect during the round
* **The privacy of the data**, the communication of the models' update have to assure a transparency on the information of client

I give you some documents to learn more about this method and its challenges

* [A comic](https://federated.withgoogle.com/) by Google
* Peter Kairouz, H. Brendan McMahan, Brendan Avent (2019). [Advances and Open Problems in Federated Learning](https://arxiv.org/abs/1912.04977).


### Frameworks

In this document we will compare the following open-source FL frameworks:
* Tensoflow Federated (TFF) of Google Inc
* Paddle Federated Learning (PaddleFL) of Baidu
* Pysift of the open community OpenMined
<!-- * IBM Federated Learning of IBM -->

Their performance are evaluate on two data sets:
* The MNIST's data set that contain images of digits
* The weather data in France, it corresponding at time-series data.

These two data set are imported in [data](/data)

In this project these frameworks are evaluate in many aspects like:

* Simplicity of the installation
* Maturity, documentation and community
* Comparison of the built model’s accuracy and of the training process performance
* Capacity to do simulation and to deployed

> My device is an Hp EliteBook, without GPU, composed by an Intel Core i7-6600U, 2.60 GHz 2.81 GHz and 16 Go of RAM


#### Tensorflow Federated of Google Inc

[Tensorflow Federated](https://github.com/tensorflow/federated) is an open source framework, developed by Google Inc. Based on the framework Tensorflow, it gives the possibility to simulate a Federated Learning. Tensorflow Federated allows to developed aggregates or learning methods. Tensoflow federated is composed by two layers:

* Federated Learning (FL) API, this layer is the high-level of the framework. It allows to do federated tranning and evaluation, with the utils give by the API 
* Federated Core (FC) API, this layer allows to test and creat new federated algorithm based on TensorFlow

Tensorflow Federated don't offer the privacy of data and the use of GPU. To add the privacy of data we can use [TensorFlowPrivacy](https://github.com/tensorflow/privacy) that allow to train machine-learning models with privacy 


#### Pysift of the open community OpenMined

[TODO]

#### PaddleFL of Baidu

[Paddle Federated learning](https://github.com/PaddlePaddle/PaddleFL) is developed by Baidu, a Chinese Company. Based on the framework PaddlePaddle, it gives the possibility to do FL with remote data on GPU or not and with some technical like Federated Averaging, Differential Privacy and Secure Aggregation. In this framework, Baidu offers two packages, **paddle_fl** that allows to work with horizontal partitioning of data and **mpc** that allows to work with vertical partitioning of data. This two method are applicable in simulation or deployment mode.

### Comparison

We can find, in the folder of each frameworks, a Readme where I propose an analyze more complete on each frameworks.

I'm going to start my comparison with the ease of finding documentation or examples and doing the installation of these frameworks. And in a second time I'm going to compare the simplicity of madding a federated learning strategy and the different results obtain with these frameworks

It's easier to learn the federated learning with Tensorflow federated than PaddleFL due to it's the most famous with Pysyft. It has some examples or documentations in English on the Internet and its important community allows to has easy some help. Conversely, PaddleFl is easier to use with a little bases in federated learning because it has a small community and it doesn't have a lot of document or example on the internet.

Moreover, PaddleFL requires many dependencies that give the installation more difficult than Tensorflow federated who it's installed with pip, this explains why Baidu recommend using their docker.

Now that we have see that Tensorflow federated is the most intuitive, we wat to compare the simplicity of madding a federated learning strategy and the different results obtain with these frameworks.
We have two types of structure on the one hand, Tensorflow federated, who use a simple script and simulate a federated learning thanks to a list that contains the data of each client (the first element is the data of the first client ...) and it browse this list to train the model on each client. 
In an other hand PaddleFL is a method closer to a deployment strategy. It composes by four scripts :
* The master that describes the FL strategy

* The scheduler that manages the exchange between the server and the clients

* The server

* The client that loads their data  and describes their trainning strategie

You can find, in the folder of each frameworks, the script to do a federated method on MNIST data set. We will compare now, the resulte of the experiments.

<table>
    <thead>
        <tr>
            <th colspan=13>Results</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <th >Model</th>
            <th colspan=3>PaddleFL with Federated Averaging</th>
            <th colspan=3>PaddlePaddle</th>
            <th colspan=3>TFF with Federated Averaging</th>
            <th colspan=3>Tensorflow</th>
        </tr>
        <tr>
            <th></th>
            <th>Accuracy</th>
            <th>Times</th>
            <th>Rounds</th>
            <th>Accuracy</th>
            <th>Times</th>
            <th>Epochs</th>
            <th>Accuracy</th>
            <th>Times</th>
            <th>Rounds</th>
            <th>Accuracy</th>
            <th>Times</th>
            <th>Epochs</th>
        </tr>
        <tr>
            <th>Multi-layer perceptron</th>
            <th>0.9749</th>
            <th>1595s (00:26:35)</th>
            <th>~ 50</th>
            <th>0.980</th>
            <th>964s (00:17:04)</th>
            <th>~ 20</th>
            <th>0.975</th>
            <th>900s (00:15:00)</th>
            <th>~ 105</th>
            <th>0.979</th>
            <th>66s (00:01:00)</th>
            <th>~ 15</th>
        </tr>
        <tr>
            <th>Convolutional NN</th>
            <th>0.9821</th>
            <th>4004s (01:06:44)</th>
            <th>~ 30</th>
            <th>0.9847</th>
            <th>1069s (00:17:49)</th>
            <th>~ 10</th>
            <th>0.980</th>
            <th>7320s (02:20:00)</th>
            <th>~ 100</th>
            <th>0.984</th>
            <th>480s (00:08:00)</th>
            <th>~ 15</th>
        </tr>
    </tbody>
</table>

This table demonstrate that in general the run takes more time when we use the federated learning, but it doesn't have a big impact on the accuracy of the model.

The cause of this slowdown is the communication time between the server and the workers or the number of cycle more important. So it's important to choose a good compromise between the number of cycle and the number of model parameters.

Tensorflow federated don't have communication time because it offers just the possibility to made simulation of federated learning. But, we can see that it takes more time and more round to converge with the CNN.

Principaly, PaddleFL offer a learning with less epoch and round, but we can see on the Multi-layer perceptron that the time of communication have an impact on the time of convergence (more time for less rounds). But, it's not the same for the CNN, due to PaddleFL run the script of clients in parallel, we can see that it takes less times than Tensorflow Federated.

### Conclusion

These are two different framework. Tensorflow federated is easy to use because it's easy to find some help, example or document in English. But, it is a research framework that offer the possibility to do some experiment like to build your personal aggregate method, but it can't be used in deployment.

In an other hand, PaddleFL is more oriented towards a deployment strategy, where each client are are connected with an IP address to the server, and where data can be in remote. But, the main problem of this framework is the lack of document or example that who gives the use more difficult.

To conclude, the accuracies obtain with these frameworks are the same, but the time of convergence is variable, PaddleFL is better than Tensorflow federated for the CNN, but not for the Multilayer perceptron. So the choice of the framework depends on the application that your desire do.

