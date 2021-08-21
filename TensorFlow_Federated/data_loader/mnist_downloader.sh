#!/bin/bash

wget https://data.deepai.org/mnist.zip 

apt-get install unzip

unzip mnist.zip -d mnist_data

rm mnist.zip

cd mnist_data

gunzip t*-ubyte.gz