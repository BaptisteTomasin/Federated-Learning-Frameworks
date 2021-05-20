from os import listdir, environ, path
from os.path import isfile, join, exists
import argparse

environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Delete warning if you don't have GPU

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def dir_path(string):
    '''
    Function to check the path
    '''
    if path.exists(string):
        return string
    else:
        raise NotADirectoryError(string)

def generateSummary(log_fh, file_name):
    '''
    Function to generate the tensoflow summary for tensorboard
    '''
    Summary_writer = tf.summary.create_file_writer(Summarydir + "/train/" + file_name[:-4])
    with Summary_writer.as_default():
        for line in log_fh:
            line_split = line.split(" ")
            round_num = int(line_split[3])
            
            if "Test" in line:
                tf.summary.scalar("Train loss", float(line_split[6]), step=round_num)
                tf.summary.scalar("Train accuracy", float(line_split[9]), step=round_num)
                tf.summary.scalar("Test loss", float(line_split[12]), step=round_num)
                tf.summary.scalar("Test accuracy", float(line_split[15][:-2]), step=round_num)
            else:
                tf.summary.scalar("Train loss", float(line_split[6]), step=round_num)
                tf.summary.scalar("Train accuracy", float(line_split[9][:-2]), step=round_num)



if __name__ == "__main__":

    # Args
    ###################

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=dir_path)
    args = parser.parse_args()

    Summarydir = "summary_logs/"

    if ".l" in args.path:
        with open(args.path) as f:
            generateSummary(f, args.path)

    else:
        for file_name in listdir(args.path)[3:]:
            with open(join(args.path, file_name)) as f:
                generateSummary(f, file_name)


