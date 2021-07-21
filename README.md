# Federated Learning (FL) Frameworks

The Machine Learning represent the fact to offer at the computer the capacity to "learn" from data. We can separate it in two parties. In a first time we perform the model from learning data, that we know and that we can analyze it to select the more adequate model for our task. The second task is the deployment of the model, when our model is optimized and offered the best score on the validation data, we can use it on production.

At this time, the learning data are collected and centralized by companies, but this approached is not always applicable. In sensitives sectors, like healthcare or finance, data are protected. To remedy this problem, the Federated Learning is a new approach of learning in development.

In this Github we discover applications of different Federated Learning frameworks like Tensorflow Federated and PaddleFL.

Before beginning to discuss these frameworks, I want to present quickly the idea of the Federated Learning.

### Federated learning main idea and documents

The main idea of Federated Learning is to share the model to the device instead of sending the data of devices to the server. The traditional approach is composed by one server and multiple clients (i.e. workers). The learning method is an iterative method where each iteration, named a round or a cycle, is composed by four step:

* Step 1: The model is sent to the clients eligible for the round.

* Step 2: Each client locally computes the gradient and updates the model with its data.

* Step 3: Local parameters are sand to the server, using encrypted communication that assure the privacy of data.

* Step 4: The server aggregates the model to design a new global model with all models trained locally.

Here we read that there is only one round, we can iterated these steps until convergence.

After this learning step, like with a centralize learning, the server sends the global model to the clients that employ it, this is the deployment task. The derivative of this approach is without the server, local parameters are shared with local aggregation.

<p float="left", style="text-align: center;">
  <img src="/images/FL_schema.png" width="600"/> 
</p>



The recent adoption of law like the GDPR in European Union, the CCPA in the USA or the PDPA in Singapore, requests transparency on personal data. Or, with the federated learning personal data remain local so with the encoding of the local model before the send data stays private this allows to respect these laws. Also, the transparency on personal data allows to gain the trust of the user and so a better relation between the company and the user.
The federated learning has other benefits. Indeed, the model is learned on the clients' device that allows the company to perform new models without a extensive data center and big computational resources. Consequently, it offers the possibility of doing machine learning to more companies.

But, federated learning remain a recent concept and is in development. Basically, the principal challenges are:

* **The communication between sever and clients**, the technics to reduce the times of communication touch many parameters like the number of rounds, the number of epochs the complexity of the model...
* **The variability of the clients**, all clients are not always open to participate at the round or also a client can be disconnected during the round
* **The privacy of the data**, the communication of the models' update have to assure a transparency on the information of the clients

You can discover some documents that explain the different challenges of the federated learning, like this document:

* [A comic](https://federated.withgoogle.com/) by Google
* Peter Kairouz, H. Brendan McMahan, Brendan Avent (2019). [Advances and Open Problems in Federated Learning](https://arxiv.org/abs/1912.04977).


### Frameworks

In this document we will compare open-source federated learning frameworks:
* Tensoflow Federated (TFF) by Google Inc
* Paddle Federated Learning (PaddleFL) by Baidu
<!-- * Pysyft of the open community OpenMined -->
<!-- * IBM Federated Learning of IBM -->

Their performances are evaluated on two datasets. The first is the MNIST data set that contains images of digits and the second is the Arma series, that corresponding at time-series data.

These two datasets are imported in [data](/data)

In this project the frameworks are evaluated in many aspects like:

* Simplicity of the installation
* Maturity, documentation and community
* Comparison of the built model’s accuracy and the training process performance
* Capacity to do studies and deployments

The community around the federated learning increased these last years. You can find other open-source framework like FATE by Webank's AI Department, Flowers, etc. that emerge but their are not part of the scope of this study.

> My device is an Hp EliteBook, without GPU, composed by an Intel Core i7-6600U, 2.60 GHz 2.81 GHz and 16 Go of RAM


#### Tensorflow Federated of Google Inc

[Tensorflow Federated](https://github.com/tensorflow/federated) is an open-source framework, developed by Google Inc. Based on Tensorflow, it offers the possibility to simulate a Federated Learning strategy. Tensorflow Federated allows to developed aggregates or learning methods. It is composed by two layers:

* Federated Learning (FL) API, this layer is the high-level of the framework. It allows doing federated training and evaluation, with the utils given by the API 
* Federated Core (FC) API, this layer allows to test and invent new federated algorithms based on TensorFlow

Tensorflow Federated don't offer the use of GPU or also the privacy of data. To add the privacy of data, we can add [TensorFlowPrivacy](https://github.com/tensorflow/privacy) that allow to train machine-learning models with privacy. 


<!-- #### Pysyft of the open community OpenMined

[Pysyft](https://github.com/OpenMined/PySyft) is developed by the open community OpenMined. It allows to perform Federated Learning within the main Deep Learning frameworks like PyTorch, Keras and TensorFlow. It combines federated learning, secured multiple-party computations and differential privacy to train robust privacy neural networks. With **Duet** it allows a data scientist to perform a model on remote data in collaboration with an owner. And in another hand, with **Pygrid** it wants to offer a deployment module of centralized or decentralized federated learning. -->

#### PaddleFL of Baidu

[Paddle Federated learning](https://github.com/PaddlePaddle/PaddleFL) is developed by Baidu, a Chinese Company. Based on PaddlePaddle, it offers the possibility to do FL with remote data on GPU or not and with some technical like Federated Averaging, Differential Privacy and Secure Aggregation. In this framework, Baidu offers two packages, **paddle_fl** that allows to work with horizontal partitioning of data and **mpc** that allows to work with vertical partitioning of data. These two methods are applicable in simulation or deployment mode.

### Comparison

We can find, in the folder of each framework, a Readme where I propose an analyze more complete on each framework.

I'm going starting my comparison with the ease of finding documentation or examples and doing the installation of these frameworks. In a second time I'm going comparing the simplicity of performing a federated learning strategy, and the different results get with these frameworks.

It's easier to learn the federated learning with Tensorflow federated than PaddleFL due to it's the most famous with Pysyft. It has some examples or documentation in English on Internet and its important community allows to have easy some help. Conversely, PaddleFl is easier to use with a little base in federated learning. In fact, it has a small community which implies it doesn't have many documents or examples on the internet.

At that, PaddleFL need many dependencies that give the installation more difficult than Tensorflow federated who it's installed with pip, this explains why Baidu recommends employing their docker.

Now that we have seen that Tensorflow federated is the most intuitive, we want to compare the simplicity of madding a federated learning strategy and the diverse results.
We differentiate two types of structure. On the one hand, Tensorflow federated, who use a simple script and simulate a federated learning thanks to a list that contains the data of each client (the first element represent the data of the first client ...) and it browse this list to train the model on each client.
In another hand PaddleFL is a method closer to a deployment strategy. It is composed by four scripts:
* The master that describes the FL strategy

* The scheduler that manages the exchange between the server and the clients

* The server

* The client that loads their data and describes their training strategy

You can find, in the folder of each framework, the script to perform a federated method on MNIST dataset. Now, we will compare, the results of the experiments.

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

This table demonstrates that in general the run requires more time when we use the federated learning, but it doesn't have a big impact on the accuracy of the model.

The cause of this slowdown is the communication time between the server and the workers or the number of cycles more important. So it's important to choose a good compromise between the number of cycles and the number of model parameters.

Tensorflow federated don't have communication time because it offers just the possibility to made simulation of federated learning. But, it doesn't parallelize the learning task, so we can see it takes more time and more round to converge with the CNN.

Mainly, PaddleFL offer a learning with fewer epochs and rounds, but we can see on the Multi-layer perceptron that the time of communication have an impact on the time of convergence (more time for fewer rounds).  But, it's not the same for the CNN, due to PaddleFL run the script of clients in parallel, we can see it takes fewer times than Tensorflow Federated.

### Conclusion

These are two different frameworks. Tensorflow federated is easy to use because it's easy to find some help, example or document in English. But, it is a research framework that offer the possibility to develop some experiment like to build your personal aggregate method, but it can't be used in deployment.

In another hand, PaddleFL is more oriented towards a deployment strategy. Because each client is connected with an IP address to the server and data can be in remote. But, the main problem of this framework is the lack of document or example that who gives the use more difficult.

To conclude, the accuracies get with these frameworks are the same but the time of convergence is variable. PaddleFL is better than Tensorflow federated for the CNN, but not for the Multilayer perceptron. So the choice of the framework depends on the application that your desire produces.

