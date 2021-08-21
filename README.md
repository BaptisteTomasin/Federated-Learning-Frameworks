# Federated Learning (FL) Frameworks

This Github explain quickly the main idea of the Federated learning and present applications of different Federated Learning frameworks like [Tensorflow Federated](https://github.com/tensorflow/federated) and [Paddle Federated learning](https://github.com/PaddlePaddle/PaddleFL).
In the end, base on the experiment, a comparison of this framework is done about the ease to use these and their performances.

### Federated learning main idea and documents

The main idea of Federated Learning is to share the model to the device instead of sending the data of devices to the server. The traditional approach is composed by one server and multiple clients (i.e. workers). The learning method is an iterative method where each iteration, named a round or a cycle, is composed by four steps:

* Step 1: The model is sent to the clients eligible for the round.

* Step 2: Each client locally computes the gradient and updates the model with its data.

* Step 3: Local parameters are sand to the server, using encrypted communication that assure the privacy of data.

* Step 4: The server aggregates the model to design a new global model with all models trained locally.

Here we read that there is only one round, we can iterate these steps until convergence.

After this learning step, like with a centralized learning, the server sends the global model to the clients that employ it, this is the deployment task. The derivative of this approach is without the server, local parameters are shared with local aggregation.

<p float="left", style="text-align: center;">
  <img src="/images/FL_schema.PNG" width="600"/> 
</p>

The recent adoption of law like the GDPR in European Union, the CCPA in the USA or the PDPA in Singapore, requests transparency on personal data. Or, with the federated learning personal data remain local so with the encoding of the local model before sending, data stays private this allows to respect these laws. Also, the transparency on personal data allows to gain the trust of the user and so a better relation between the company and the user.
The federated learning has other benefits. Indeed, the model is learned on the clients' device that allows the company to perform new models without an extensive data center and big computational resources. Consequently, it eases the use of machine learning for companies.

But, federated learning remain a recent concept and is in development. Basically, the principal challenges are:

* **The communication between server and clients**, the technics to reduce the times of communication touch many parameters like the number of rounds, the number of epochs the complexity of the model...
* **The variability of the clients**, all clients are not always open to participate in the round or also a client can be disconnected during the round.
* **The privacy of the data**, the communication of the models' update have to assure a transparency on the information of the clients.

You can discover some documents that explain the different challenges of the federated learning, like this document:

<a id="1">[1]</a> [A comic by Google](https://federated.withgoogle.com/).

<a id="2">[2]</a> Peter Kairouz, H. Brendan McMahan, Brendan Avent (2019). [Advances and Open Problems in Federated Learning](https://arxiv.org/abs/1912.04977).

### Study presentation

In this document we will compare open-source federated learning frameworks:
* Tensoflow Federated (TFF) by Google Inc
* Paddle Federated Learning (PaddleFL) by Baidu

Their performances are evaluated on two datasets. The first is the MNIST data set that contains images of digits and the second is the Arma series, that corresponding at time-series data. The Arma dataset is present in the folder [data](/data) and the experience can be done with TFF (see the example of config file in the [config_examples](TensorFlow_Federated/config_examples) folder) for paddleFL it will be done in a next update

In this project the frameworks are evaluated in many aspects like:

* Simplicity of the installation.
* Maturity, documentation and community.
* Comparison of the built model’s accuracy and the training process performance.
* Capacity to do studies and deployments.

The community around the federated learning increased these last years. You can find other open-source framework like FATE by Webank's AI Department, Flowers, etc. that emerge but their are not part of the scope of this study.

><table>
>    <thead>
>        <tr>
>            <th colspan=2>Computer config</th>
>        </tr>
>    </thead>
>    <tbody>
>        <tr>
>            <th >Model</th>
>            <th >HP Z440 Workstation</th>
>        </tr>
>        <tr>
>            <th>GPU</th>
>            <th>No</th>
>        </tr>
>        <tr>
>            <th>Processor</th>
>            <th>Intel(R) Xeon(R) CPU E5-1620 v4 @ 3.50GHz</th>
>        </tr>
>        <tr>
>            <th>RAM</th>
>            <th>64 GiB</th>
>        </tr>
>        <tr>
>            <th>Storage</th>
>            <th>1 TB</th>
>        </tr>
>    </tbody>
></table>

#### Tensorflow Federated of Google Inc

[Tensorflow Federated](https://github.com/tensorflow/federated) (TFF) is an open-source framework, developed by Google Inc. Based on the framework Tensorflow, it allows to simulate a Federated Learning strategy. Tensorflow Federated allows to developed aggregates or learning methods. It is composed by two layers:

* Federated Learning (FL) API, this layer is the high-level of the framework. It allows doing federated training and evaluation, with the tools give by the API.
* Federated Core (FC) API, this layer allows to test and devise new federated algorithms based on TensorFlow.

It doesn't offer the privacy of data and the use of GPU. But [TensorFlowPrivacy](https://github.com/tensorflow/privacy) can be adding, that allow to train machine-learning models with privacy. 

#### PaddleFL of Baidu

[Paddle Federated learning](https://github.com/PaddlePaddle/PaddleFL) (PaddleFL) is developed by Baidu, a Chinese Company. Based on the framework PaddlePaddle, it offers the possibility to do Federated Learning with remote data. With the possibility to use GPU or not, it has some technical like Federated Averaging, Differential Privacy and Secure Aggregation. In this framework, Baidu offers two packages, **paddle_fl** that allows to work with horizontal partitioning of data and **mpc** that allows to work with vertical partitioning of data. These two methods are applicable in simulation or deployment mode.

### Framework comparison

In each framework folder an analysis more complete are proposed.
 
Firstly a comparison of these frameworks are done in the following table (documentation, examples, the ease to install it...). Secondly we will compare the simplicity of performing a federated learning strategy, and the different results get with these frameworks.


<p float="left", style="text-align: center;">
  <img src="/images/framework_comparison.PNG"/> 
</p>

Based on facts, Tensorflow federated is more used than PaddleFL which explain the number of example on Internet more important that makes learning of the FL bases faster with TFF.

Two different types of code structure are presented with these two frameworks. On the one hand, Tensorflow federated, who uses a simple script and simulates a federated learning thanks to a list that contains the data of each client (the first element represent the data of the first client ...) and it browses this list to train the model on each client.

On the other hand, PaddleFL is a method closer to a deployment strategy. It is composed of four scripts:
* The master that describes the FL strategy.

* The scheduler that manages the exchange between the server and the clients.

* The server that made the aggregation.

* The client that loads their data and describes their training strategy.

You can find, in the folder of each framework, the script to perform a federated method on MNIST and Arma dataset. Now, we will compare, the results of the experiments.

For the experience, the strategy is composed by 10 clients that train the model locally on 3 epochs by round. The data of the client are randomly selected on all the train dataset of the MNIST, the distribution of the data, used in my study, is done in the [data](/data) folder. For the training, we will use the SGD optimizer with a learning rate of 0.1. To compare the accuracy of the model, obtains with the centralized and the federated strategies, we test the model on the test dataset of MNIST. 

<p float="left", style="text-align: center;">
  <img src="/images/mnist_results.PNG"/> 
</p>

As a reminder, an epoch corresponds to the reading of all the learning examples. In the case of the federated network, a global epoch is realized when each client has realized an epoch which corresponds to the number of rounds x the number of epochs per client.

**Notes:**

* With Federated Averaging, the accuracies is not impacted.
* The run time is more important when we use the federated learning. Due to the communication time between the server and the workers or the number of loops more important.
* It's important to choose a good compromise between the number of cycles and the number of model parameters, like number of rounds and number of local epochs
* Tensorflow Federated take more time than PaddleFL with big model, because it doesn't parallelize the local training.
* Mainly, PaddleFL offer a learning with fewer epochs and rounds
* The time of communication have an impact on the time of convergence. The Multi-layer perceptron take more time for fewer rounds with PaddleFL (use communication) than TFF (no communication)

### Conclusion

These are two different frameworks. Tensorflow federated is easy to use because it's easy to find some help, example or document in English. But, it is a research framework that offer the possibility to develop some experiment like to build your personal aggregate method, but it can't be used in deployment.

On the other hand, PaddleFL is more oriented towards a deployment strategy. Because each client is connected with an IP address to the server and data can be in remote. But, the main problem of this framework is the lack of document or example that who gives the use more difficult.

The accuracies get with these frameworks are the same, but the time of convergence is variable. 

So the choice of the framework depends on the application that your desire produces.

