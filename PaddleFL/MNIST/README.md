# MNIST data set 

In this folder, we run a federated learning script with PaddleFL. Moreover, in the script [paddlepaddle_mninst.py](/PaddleFL/MNIST/paddlepaddle_mninst.py), you can find the script that run centralize learning on the same data set with paddelpaddle.

### Pre-requist 

To work with the mnist data set, downloaded in the folder [data](/data), you must to install **mlxtend**, with the following command:

    pip3 install mlxtend

### Run script

To begin the experiment we have to generate the **data_idx.json** file. If it is not already been done, you can you refer to the [data](/data) folder.

Now that we have the distribution of data, we can run and stop the federated learning script with the following commands:

    # Run the FL
    ./run.sh
    # Stop the FL
    ./stop.sh

To conclude, you can run the paddlepaddle's script with this command, to compare your results:

    python3 paddlepaddle_mninst.py

> Remarks: 
> * For the federated program, you can see the test accuracy in the log of the client 0
> * To display the graph of metrics for each clients and for test data, you can use [parser.py](/PaddleFL/MNIST/parser.py) with the following command: 
>
>       python3 parser.py --path <directory to the log file or the log folder>
>
> That return the folder named **summary_logs**. After this you can run Tensorboard with this command:
>
>       tensorboard --logdir \<directory to summary_logs \>
>
> For example, with the CNN, I obtain this graph:
><table>
    <tr>
        <th colspan=2>CNN PaddleFL</th>
    </tr>
  <tr>
    <td>Train Loss</td>
     <td>Train Accuracy</td>
  </tr>
  <tr>
    <td><img src="../../images/paddlefl_MNIST_CNN_train_loss.png" width=300></td>
    <td><img src="../../images/paddlefl_MNIST_CNN_train_acc.png" width=300></td>
  </tr>
  <tr>
    <td>Test Loss</td>
     <td>Test Accuracy</td>
  </tr>
  <tr>
    <td><img src="../../images/paddlefl_MNIST_CNN_test_loss.png" width=300></td>
    <td><img src="../../images/paddlefl_MNIST_CNN_test_acc.png" width=300></td>
  </tr>
 </table>
<table>
    <tr>
        <th colspan=2>CNN PaddlePaddle</th>
    </tr>
  <tr>
    <td>Train Loss</td>
     <td>Train Accuracy</td>
  </tr>
  <tr>
    <td><img src="../../images/paddlepaddle_MNIST_CNN_train_loss.png" width=300></td>
    <td><img src="../../images/paddlepaddle_MNIST_CNN_train_acc.png" width=300></td>
  </tr>
  <tr>
    <td>Test Loss</td>
     <td>Test Accuracy</td>
  </tr>
  <tr>
    <td><img src="../../images/paddlepaddle_MNIST_CNN_test_loss.png" width=300></td>
    <td><img src="../../images/paddlepaddle_MNIST_CNN_test_acc.png" width=300></td>
  </tr>
 </table>