# PaddleFL of Baidu

[Paddle Federated learning](https://github.com/PaddlePaddle/PaddleFL) is developed by Baidu, a Chinese Company. Based on the framework PaddlePaddle, it gives the possibility to do FL with remote data on GPU or not and with some technical like Federated Averaging, Differential Privacy and Secure Aggregation. In this framework, Baidu offers two packages, **paddle_fl** that allows to work with horizontal partitioning of data and **mpc** that allows to work with vertical partitioning of data. This two method are applicable in simulation or deployment mode.

In [MNIST](/PaddleFL/MNIST/) you can find the application of PaddleFL on the MNIST's data set.

### Work environment

To beginning we will create our work environment. It's recommended using the docker image provide by Baidu, you can see the next part to view how to work with it.

Its size is 17giga. If it's a problem for you an alternative of this docker image it's to install it from scratch, there takes some difficulty with dependencies. The link to the instruction of installation can be found on the documentation, at this [link](https://paddlefl.readthedocs.io/en/latest/compile_and_intall.html#compile-from-source-code)

#### Docker image

If you desire test PaddleFL to see if this framework is interesting, I recommend using the docker image and so you avoid some problem with dependencies. When you have installed [Docker](https://www.docker.com/), the shell's command to download the docker's image of Baidu is the following:

    docker pull paddlepaddle/paddlefl:latest

This docker image allow you to run all PaddleFL's script. For this, nothing could be easier, you run the docker image with the following command:

    docker run -it --rm -v <directory of the paddleFL's project>:/My_program/ -v <directory of the data folder>:/data/ paddlepaddle/paddlefl

After it, your project is in the folder **My_program** of the docker and your data is in the folder **data** of the docker. To take more information about the use of Docker, you can see this [tutorial](https://docs.docker.com/get-started/).

> If you use WSL2, PaddleFL has a problem of compatibility with it kernel. To resolve it, in the folder of the user (for me C:\Users\ <user_name>), you have to create a new file, named **.wslconfig**, who contain the following lines:

    [wsl2]
    kernelCommandLine = vsyscall=emulate

### Analyse of this framework


Currently, PaddleFL has small community with 66 issues and 12 contributor on GitHub. This explain the difficulty to find examples of script. The lack of documentation in English (the majority of articles are in Chinese), the few commentary in the script, the few examples of script and the documentation not detailed makes it difficult to work with this framework.

To evaluate this framework, I appliqued PaddleFL on the MNIST data set. You can see the commented script in the folder [MNIST](/PaddleFL/MNIST/). For my experience I use ten clients and I do three epochs by round. The data of client is randomly selected, on all the train data set of MNIST, the distribution of data is done in the folder [data](/data). And to compare the accuracy of the model we test it on the test data set of MNIST.

Two models are proposed:
* A multilayer perceptron compose by two fully connected layer their size is 100, with the ReLU's activation function, and the fully connected output layer, the size is the label size (10) with the softmax's activation function.
* A convolutional neural network, compose by:
    * A first convolutional layer, the convolution kernel size is 3*3, a total of 32 convolution kernels
    * A first pooling layer, the pooling size is 2*2, the step size is 1, and the maximum pooling
    * A flatten layer
    * A first fully connected layer, with size 128
    * A fully connected output layer, the size is the label size (10) and with softmax as the activation function

All this models are defined in the file [models.py](/PaddleFL/MNIST/models.py)

In the following, a summary table of all my accuracies obtain on the test's data set.

<table>
    <thead>
        <tr>
            <th colspan=3>Accuracy</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <th>Model</th>
            <th>PaddleFL with Federated Averaging</th>
            <th>PaddlePaddle</th>
        </tr>
        <tr>
            <th>Multi-layer perceptron</th>
            <td>0.9749 (~ 50 rounds)</td>
            <td>0.980 (~ 20 epochs)</td>
        </tr>
        <tr>
            <th> Convolutional NN </th>
            <td>0.9821 (~ 30 rounds)</td>
            <td>0.9847 (~ 10 epochs)</td>
        </tr>
    </tbody>
</table>

This table shows us that with the method of Federated Averaging, the Federated learning don't have a big impact on the accuracy of the model especially if it's a complex neural network. For example, the convolutional neural network give us the same result. A second aspect interesting to see, is the time of convergences.

<table>
    <thead>
        <tr>
            <th colspan=3>Times</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <th>Model</th>
            <th>PaddleFL with Federated Averaging</th>
            <th>PaddlePaddle</th>
        </tr>
        <tr>
            <th>Multi-layer perceptron</th>
            <td>1595s (00:26:35)</td>
            <td>964s (00:17:04)</td>
        </tr>
        <tr>
            <th> Convolutional NN </th>
            <td>4004s (01:06:44) </td>
            <td>1069s (00:17:49)</td>
        </tr>
    </tbody>
</table>

This second table, more interresting, demonstrate that the run time is more important when we use the federated learning. The cause of this slowdown is the communication time between the server and the workers or the number of loop more important. You can check the increase of times with the CNN (an important number of parameters) where the time for execution is four times more important and the number of rounds is three time more important than the number of epochs.

We can say that this framework offers many interesting strategy of FL like Federated Averaging, Differential Privacy and Secure Aggregation. But it lacks maturity and it is felt with the  lack of example or documentation. It can be a good framework because it offers a large variety of tools, in simulation and deployment, but at this time, Baidu don't have updated this tool since last 11 month. I think, we have to wait 1 or 2 years to see if it's evolved, if the community increases, if Baidu developed more examples and suggest better documentation.