# Pysyft of the open community OpenMined


[Pysyft](https://github.com/OpenMined/PySyft) is developed by the open community OpenMined. It allows to perform Federated Learning within the main Deep Learning frameworks like PyTorch, Keras and TensorFlow. It combines federated learning, secured multiple-party computations and differential privacy to train robust privacy neural networks. With **Duet** it allows a data scientist to perform a model on remote data in collaboration with an owner. And in an other hand, with **Pygrid** it wants to offer a deployment module of centralized or decentralized federated learning.

In [MNIST](/Pysyft/MNIST/) you can find the application of Pysyft on the MNIST's dataset.

### Work environment

To beginning we will create our work environment. 
If you are on Ubuntu or MacOS, you can install it with the classicals instructions.

Else you can also install [Docker](https://www.docker.com/), and after, to build the image with this Dockerfile the command is the following:

    docker build -t project/pysyft <directory to the Dockerfile> --build-arg CUDA=<"True" or "False">

This docker image allow you to run all Pysyft Federated's script. For this, nothing could be easier, you run the docker image with the following command:

    docker run -it --rm -v <directory of the Pysyft's project>:/My_program/ -v <directory of the data folder>:/data/ project/pysyft /bin/sh

After it, your project is in the folder **My_program** of the docker and your data is in the folder **data** of the docker. To take more information about the use of Docker, you can see this [tutorial](https://docs.docker.com/get-started/)