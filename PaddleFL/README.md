# PaddleFL of Baidu

[Paddle Federated learning](https://github.com/PaddlePaddle/PaddleFL) (PaddleFL) is developed by Baidu, a Chinese Company.  Based on the framework PaddlePaddle, it gives the possibility to do FL with remote data. With the possibility to use GPU or not, it has some methods like Federated Averaging, Differential Privacy and Secure Aggregation. In this framework, Baidu offers two packages, **paddle_fl** that allows to work with horizontal partitioning of data and **mpc** that allows to work with vertical partitioning of data. These two methods are applicable on a single computer to do simulation or to do deployment on multiple device.

### Work environment

To begin, we will establish our work environment. It's recommended using the docker image provide by Baidu, you can see in the next part how work with it.

Its size is 17GB. If it's a problem for you, an alternative of this docker image is to install it from scratch, there takes some difficulty with dependencies. The link to the instruction of installation can be found on the documentation, at this [link](https://paddlefl.readthedocs.io/en/latest/compile_and_intall.html#compile-from-source-code).

#### Docker image

It is recommended to use the docker image provided by Baidu and so you eliminate some problem with dependencies during the installation. When you have installed [Docker](https://www.docker.com/), the shell's command to download the docker's image of Baidu is:

    docker pull paddlepaddle/paddlefl:latest

This docker image allows you to execute all PaddleFL's script. For this, nothing could be easier, you run the docker image with the following command:

    docker run -it --rm -v <PROJECT PATH>:/My_program/ -v <DATA PATH>:/data/ paddlepaddle/paddlefl

After two folders are created in the docker:
    * **My_program** that contains the project code. 
    * **data**, that contains the data.
To acquire more information about the use of Docker, you can observe this [tutorial](https://docs.docker.com/get-started/).

> If you use WSL2, PaddleFL has a problem of compatibility with it kernel. 
> To resolve it, in the folder of the user (for me C:\Users\ <USER_NAME>), you have to create a new file, named **.wslconfig**, who contain the following lines:
>
>       [wsl2]
>       kernelCommandLine = vsyscall=emulate

### Experiment execution

To begin the experiment, personalize the [config.json](/PaddleFL/config.json) file and install the requirement with this command:

    pip3 install -r requirement

Now that the configs files are edited, perform the strategy with the following command:

    sh run.sh <PATH TO THE CONFIG FILE>

> Notes:
>   * Some examples of configs file are gave in the folder [configs_examples](/PaddleFL/configs_examples)
>   * To use personal data and models, a dummy file is present in [data_loader](/PaddleFL/data_loader) and [models](/PaddleFL/models) that contains the template to generate new data loader and model class.
>During the run, a result folder is created. That can be used to analyze the diverse metrics (clients and server) with Tensorboard. To display the Tensorboard interface, you can use this command:
>
>           tensorboard --logdir <PATH TO THE RESULT FOLDER\>
>
> For example, with the MP, I get this graph:
><table>
>  <tr>
>      <th colspan=2>MP PaddleFL</th>
>  </tr>
>  <tr>
>    <td>Train Loss</td>
>     <td>Train Accuracy</td>
>  </tr>
>  <tr>
>    <td><img src="../images/paddlefl_MNIST_MP_train_loss.png" height=100></td>
>    <td><img src="../images/paddlefl_MNIST_MP_train_acc.png" height=100></td>
>  </tr>
>  <tr>
>    <td>Test Loss</td>
>     <td>Test Accuracy</td>
>  </tr>
>  <tr>
>    <td><img src="../images/paddlefl_MNIST_MP_test_loss.png" height=100></td>
>    <td><img src="../images/paddlefl_MNIST_MP_test_acc.png" height=100></td>
>  </tr>
> </table>
> 
><table>
>  <tr>
>      <th colspan=2>MP PaddlePaddle</th>
>  </tr>
>  <tr>
>    <td>Train Loss</td>
>     <td>Train Accuracy</td>
>  </tr>
>  <tr>
>    <td><img src="../images/paddlepaddle_MNIST_MP_train_loss.png" height=100></td>
>    <td><img src="../images/paddlepaddle_MNIST_MP_train_acc.png" height=100></td>
>  </tr>
>  <tr>
>    <td>Test Loss</td>
>     <td>Test Accuracy</td>
>  </tr>
>  <tr>
>    <td><img src="../images/paddlepaddle_MNIST_MP_test_loss.png" height=100></td>
>    <td><img src="../images/paddlepaddle_MNIST_MP_test_acc.png" height=100></td>
>  </tr>
> </table>

### Framework analysis


Currently, PaddleFL has a weak community that generates difficulty to find some help or to find examples of script. This lack of example is the main default of the framework. The majority of documentation is in Chinese (the English documentation is not detailed and sometimes not up to date) and you find few examples or comments in the script. It is  accessible if you have already practiced the federated learning with another framework.

To evaluate this framework, PaddleFL are applied on the MNIST dataset. You can follow the config file in the folder [configs_examples](/PaddleFL/configs_examples/) to run the experiment. The FL strategy is composed by 10 clients that train the model locally on 3 epochs by round. The aggregation method is the FedAvg classical method because it is the unique common method between PaddleFl and TFF. For the training, we will use the SGD optimizer with a learning rate of 0.1. The data of the client are randomly selected on all the train dataset of the MNIST, the distribution of the data, used in my study, is done in the [data](/data) folder. To compare the accuracy of the model, obtains with the centralized and the federated strategies, we test the model on the test dataset of MNIST. 

Two models are proposed:
* A multilayer perceptron composed by:
    * Two fully connected layers (size: 100, activation function: ReLU)
    * A fully connected output layer (size: 10, activation function: Softmax)
* A convolutional neural network, compose by:
    * A first convolutional layer (kernel size: 3*3, sub convolution kernels: 32).
    * A first max pooling layer, (size: 2*2, step size: 1).
    * A flatten layer.
    * A first fully connected layer (size: 128, activation: None).
    * A fully connected output layer (size: 10, activation function: Softmax)

All these models are defined in the [models](/PaddleFL/models) folder.

Each model is trained until the convergence or the overfitting. Two strategies are possible to select the metrics. When the model converge, metrics are picked at the first epoch/round of the convergence. And when there are overfitting,  metrics are picked at the last epoch/round before the overfitting.

In the following, a summary table of all my accuracies obtains on the test's dataset.

<p float="left", style="text-align: center;">
  <img src="/images/mnist_paddlepaddle_results.PNG"/> 
</p>

As a reminder, an epoch corresponds to the reading of all the learning examples. In the case of the federated network, a global epoch is realized when each client has realized an epoch which corresponds to multiply the number of rounds by the number of epochs per client.

**Notes:**

* With Federated Averaging, the accuracies is not impacted.
* The runtime is more important when we use the federated learning. Due to the communication time between the server and the workers or the higher number of iteration.
* Small model take more loops to converge, so more communication between the server and the workers that increase the run time.
* Due to the parallelization of the local training and the small increase in the number of loops, the CNN does not take much extra time during the federated training compared to the centralized training. The extra time is the communication time.

This framework offers many interesting strategies of FL like Federated Averaging, Differential Privacy and Secure Aggregation and the possibility to made simulation or deployment. But we have to wait some times to detect if it's evolved, if the community increases, if more examples are developed and if an English documentation is published.


### Little Conclusion

PaddleFL is more oriented towards a deployment strategy. Each client is connected with an IP address to the server and data can be in remote. It can also be used in simulation mode, by launching on the same machine the python script of the different clients which are then executed in parallel (like the experiment). Moreover, if we want to simulate a FLstrategy with a lot of clients (50 clients), this framework requires good computational resources.