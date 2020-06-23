import os

os.environ["OMP_NUM_THREADS"] = "NUM_PARALLEL_EXEC_UNITS"
os.environ["KMP_BLOCKTIME"] = "30"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
import sys
if "win" in sys.platform :
    from .utils.common_layers import *
    from .utils.common_loader import *
    from .utils.train_loader import *
else :
    from utils.common_layers import *
    from utils.common_loader import *
    from utils.train_loader import *

import numpy as np
import argparse
import os
from keras import backend as K
from keras import models
import tensorflow as tf

NUM_PARALLEL_EXEC_UNITS = 10
config = tf.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS,
                        inter_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS,
                        allow_soft_placement=True,
                        device_count = {'GPU': 0, 'CPU': NUM_PARALLEL_EXEC_UNITS })

session = tf.Session(config=config)
K.set_session(session)

"""
@description：lr的infer过程



keraspython -u infer.py   \
--mean_data_file=data_both/mean_data  \
--model_file=data_train/train_result_fm/endEpoch0.h5    \
--infer_sample=data_infer/infer_sample/infer_sample_0029    \
--infer_result_dir=data_infer/infer_result_fm  \
>  data_infer/infer_result_fm/infer_log_0029   2>&1    &   
"""
def parse_args():
    parser = argparse.ArgumentParser(description="Infer process")
    parser.add_argument('--mean_data_file', type=str, default='data_both/mean_data',
                        help='Input mean data file')
    parser.add_argument('--model_file', type=str, default='',
                        help='Input file of model')
    parser.add_argument('--infer_sample', type=str, default='',
                        help='Input infer sample')
    parser.add_argument('--infer_result_dir', type=str, default='',
                        help='Input  infer resutl dir ')

    return parser.parse_args()

if __name__ == "__main__":
    print("---------------------------")
    mean_data_file = parse_args().mean_data_file
    model_file = parse_args().model_file
    infer_sample = parse_args().infer_sample
    infer_result_dir = parse_args().infer_result_dir

    os.system("echo \" start infer `date '+%Y-%m-%d %H:%M:%S'` \"")
    model = models.load_model(model_file, custom_objects={'FM': FM})
    model.summary()

    mean_data = load_mean_data(file_name=mean_data_file) # ndarray([1275,] "float32")

    writer = open(infer_result_dir + "/infer_result_"+infer_sample[-4:], "w")
    for line in open(infer_sample,"r"):
        line_list = line.strip().split(":")
        if len(line_list) != 2 :
            continue
        dnum = line_list[1].strip()
        x = line_list[0].strip().split(",")
        if len(x) != 1275 :
            continue
        x = [ float(i.strip()) for i in x]

        x = np.array(x , dtype="float32")
        x = x > mean_data
        x = np.array(x , dtype="int32") # array([1275,] , "int32")

        sample_x = []
        for i in x:
            if i == 0:
                sample_x = sample_x + [1.0 , 0.0]
            else :
                sample_x = sample_x + [0.0 , 1.0]
        sample_x = np.array([sample_x] , dtype="float32") # [1,2550]
        predict_result = model.predict(sample_x) # [1,1]

        writer.write(dnum+",%.6f,\n" % predict_result[0,0])
        writer.flush()
    writer.flush()
    writer.close()








