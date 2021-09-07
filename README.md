# Federated Learning (FL) Frameworks

This Github explain quickly the main idea of the Federated learning and present applications of different Federated Learning frameworks like [Tensorflow Federated](https://github.com/tensorflow/federated) and [Paddle Federated learning](https://github.com/PaddlePaddle/PaddleFL).
In the end, base on the experiment, a comparison of this framework is done about the ease to use these and their performances.

### Federated learning main idea and documents

The main idea is to send the model to the device. So, the data of the device is not centralized. The traditional approach is composed by one server and multiple clients (i.e. workers). The learning method is an iterative method where each iteration, named a round or a cycle, is composed by four steps:

* Step 1: The model for classification or prediction is sent to the clients eligible for the round.
* Step 2: Each client computes locally the gradient with their locally stored private data and updates the model.
* Step 3: Local parameters are sent to the server, using encrypted communication that assure the privacy of data.
* Step 4: The server aggregates the model to build a new universal model based on all models trained locally.

These steps correspond to one round, we can repeat multiple times these steps until convergence.
After this learning step, like with a centralized learning, the server sends the global model to the clients that use it, this is the deployment task. The derivative of this approach is without the server, local parameters are shared with local aggregation.

<p float="left", style="text-align: center;">
  <img src="/images/FL_schema.PNG" width="600"/> 
</p>

The recent adoption of law like the GDPR in European Union, the CCPA in  the USA or the PDPA in Singapore, requests transparency on personal data. With the federated learning personal data remain local so, with the encoding of the local model before to be send, data remains private, this allows to respect these laws. Moreover, the transparency on personal data allows to improve the user's trustworthiness and so a better relationship between the company and the user.
Federated learning has other benefits. Indeed, the model is learned on the clients' device, that allows the company to perform new model without a big data center and big computational resources. So, more companies have the possibility of doing machine learning.

But, federated learning is a new concept and is under development. Basically, the principal challenges are :

* **The communication between server and clients**, the technics to reduce the times of communication depends on many parameters like the number of rounds, the number of epochs, the complexity of the model ...

* **The variability of the clients**, all clients are not always open to participate at the round or also a client can be disconnected during the round

* **The privacy of the data**, the communication of the models' update have to assure a transparency on the information of client

Some documents explain the different challenges of the federated learning, [Advances and Open Problems in Federated Learning](https://arxiv.org/abs/1912.04977) (2019) by Peter Kairouz, H. Brendan McMahan, Brendan Avent  is the more detailed.

### Study presentation


In this report, we compare open-source federated learning frameworks:
* Tensoflow Federated (TFF) by Google Inc.
* Paddle Federated Learning (PaddleFL) by Baidu.

Their performances are evaluated on two data sets, the first is The Mixed National Institute of Standards and Technology dataset, named MNIST, that contains a group of handwritten digits where a label is associated to. The second is the Arma series, that correspond to time-series data.  The Arma dataset and an example of MNIST distribution is present in the folder [data](/data)
In this project, the frameworks are evaluated in many aspects like :

* Simplicity of the installation.
* Maturity, documentation and community.
* Comparison of the built model’s accuracy and the training process performance.
* Capacity to do studies and deployments.

The community around the federated learning increased these last years, so you can find other open-source framework like PySyft, FATE by Webank's AI Department, Flowers, etc. that emerge, but they are no part of the scope of this study.

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

[Tensorflow Federated](https://github.com/tensorflow/federated) (TFF) is an open-source framework, developed by Google Inc.Based on the framework Tensorflow, it gives the possibility to simulate a Federated Learning strategy. Tensorflow Federated allows to developed aggregates or learning methods. It is composed by two layers:

* Federated Learning (FL) API, this layer is the high-level of the framework. It allows doing federated training and evaluation, with the tools implemented by the API
* Federated Core (FC) API, this layer allows testing and creating new federated algorithm based on TensorFlow

It does not offer the privacy of data and the use of GPU. But [TensorFlowPrivacy](https://github.com/tensorflow/privacy) can be added, that allows to train machine-learning models with data privacy.

#### PaddleFL of Baidu

[Paddle Federated learning](https://github.com/PaddlePaddle/PaddleFL) (PaddleFL) is developed by Baidu, a Chinese Company.  Based on the framework PaddlePaddle, it gives the possibility to do FL with remote data. With the possibility to use GPU or not, it has some methods like Federated Averaging, Differential Privacy and Secure Aggregation. In this framework, Baidu offers two packages, **paddle_fl** that allows to work with horizontal partitioning of data and **mpc** that allows to work with vertical partitioning of data. These two methods are applicable on a single computer to do simulation or to do deployment on multiple device.

### Framework comparison

In each framework folder an analysis more complete are proposed.
 
Firstly a comparison of these frameworks are done in the following table (documentation, examples, the ease to install it...). Secondly we will compare the simplicity of performing a federated learning strategy, and the different results get with these frameworks.


<p float="left", style="text-align: center;">
  <img src="/images/framework_comparison.PNG"/> 
</p>

Based on facts, Tensorflow federated is more used than PaddleFL which explain the number of example on Internet more important that makes learning of the FL bases faster with TFF.

Two different types of code structure are presented with these two frameworks. On the one hand, Tensorflow federated, where FL environment is build by a unique script and with a list scan with a list that contains the data of each client (the first element represent the data of the first client ...), Tensorflow can be used to deploy a FL strategy, but it can be used to experiment new method and simulate strategies.

On the other hand, PaddleFL is a method closer to a deployment strategy. It is composed of four scripts:

* The master that describes the FL strategy. We define the model, the client configurations (optimizer, loss...) and the FL strategy (fed_avg, dpsgd, sec_agg). Then we used the PaddleFL's function *FL-Job-Generator* to generate processes for server and workers.

* The scheduler that manages the exchange between the server and the clients. We have to specify the scheduler IP, and we let server and clients to do registration. We can in the same time define the number of client request to do a round.

* The server that aggregate the central model. We have to specify the scheduler and server IP, and then we load and run the FL server job. It will connect to scheduler, send the model to the clients and stay the update to aggregate the global model.

* The client that train locally the model, with personal data.  We load and prepare the client's data to input it on the model. We have to specify the scheduler and server IP, the number of epoch, the number of round, and after we load and run the FL trainer job. This script return the clients' validation and test metrics and loss.

You can find, in the folder of each framework, the script to perform a federated method on MNIST and Arma dataset (just with TFF). Now, we will compare, the results of the experiments.


For my experience, we use 10 clients. We establish a common data distribution for the centralized and the federated learning, and for Tensorflow Federated and PaddleFL that allows to compare on the same base the different frameworks. The data of client are randomly selected on all the train MNIST dataset and we split the client data in a train and a validation dataset. Moreover, we establish a test dataset (i.e. the test dataset of MNIST) that allows to compare the accuracy of the model. You can fin an example an example of distribution in this [file](data/mnist_clients_distribution_10_clients.json).

The first FL strategy is composed by 10 clients that train the model locally on 3 epochs by round. The aggregation method is the FedAvg classical method because it is the unique common method between PaddleFl and TFF. For the training, we will use the SGD optimizer with a learning rate of 0.1.

Each model is trained until the convergence or the overfitting. Two strategies are possible to select the metrics. When the model converge, metrics are picked at the first epoch/round of the convergence. And when there are overfitting,  metrics are picked at the last epoch/round before the overfitting.

In the next table, we will analyze the execution time, the number of rounds and the accuracy obtains on these two models with these different frameworks.

<p float="left", style="text-align: center;">
  <img src="/images/mnist_results.PNG"/> 
</p>

As a reminder, an epoch corresponds to the reading of all the learning examples. In the case of the federated network, a global epoch is realized when each client has realized an epoch which corresponds to multiply the number of rounds by the number of epochs per client.

**Notes:**

* With Federated Averaging, the accuracies is not impacted.
* The runtime is more important when we use the federated learning. Due to the communication time between the server and the workers or the higher number of loops.
* It's important to choose a good compromise between the number of cycles and the number of model parameters, like number of rounds and number of local epochs
* Tensorflow Federated takes more time than PaddleFL with big model, because it doesn't parallelize the local training.
* PaddleFL offers a learning with fewer epochs and rounds
* The time of communication has an impact on the time of convergence. The Multi-layer perceptron takes more time for fewer rounds with PaddleFL (use communication) than TFF (no communication)


### Conclusion

These are two different frameworks. Tensorflow federated is easy to use because it's easy to find some help, example or document in English. But, it is a research framework that offer the possibility to develop some experiment like to build your personal aggregate method, but it can't be used in deployment.

On the other hand, PaddleFL is more oriented towards a deployment strategy. Because each client is connected with an IP address to the server and data can be in remote. But, the main problem of this framework is the lack of document or example that who gives the use more difficult.

The accuracies get with these frameworks are the same, but the time of convergence is variable. 

So the choice of the framework depends on the application that your desire produces.

