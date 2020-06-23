import os

from keras.layers import Activation

os.environ["OMP_NUM_THREADS"] = "NUM_PARALLEL_EXEC_UNITS"
os.environ["KMP_BLOCKTIME"] = "30"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
import sys
if "win" in sys.platform :
    from .utils.common_loader import *
    from .utils.train_loader import *
    from .utils.common_layers import *
else :
    from utils.common_loader import *
    from utils.train_loader import *
    from utils.common_layers import *

from keras import layers , models , optimizers, losses , metrics ,initializers
from keras import backend as K
import  numpy as np
import tensorflow as tf
import argparse

NUM_PARALLEL_EXEC_UNITS = 10
config = tf.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS,
                        inter_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS,
                        allow_soft_placement=True,
                        device_count = {'GPU': 0, 'CPU': NUM_PARALLEL_EXEC_UNITS })

session = tf.Session(config=config)
K.set_session(session)

"""
@description：fm在sgd优化器下的效果。

keraspython -u train_on_fm_sgd.py \
--mean_data=data_both/mean_data \
--val_sample=data_train/val_sample \
--train_sample=data_train/train_sample \
--model_result_dir=data_train/train_resultfm_sgd \
> train_fm_log_sgd 2>&1  & 
"""

def parse_args():
    parser = argparse.ArgumentParser(description=" Keras train on batch ")
    parser.add_argument('--mean_data', type=str, default='data_both/mean_data',
                        help='Input mean data ')
    parser.add_argument('--val_sample', type=str, default='data_train/val_sample',
                        help='Input val sample ')
    parser.add_argument('--train_sample', type=str, default='data_train/train_sample',
                        help='Input train sample ')
    parser.add_argument('--model_result_dir', type=str, default='data_train/train_result',
                        help='Input model result dir')
    return parser.parse_args()


if __name__ == "__main__":
    mean_file = parse_args().mean_data
    val_file = parse_args().val_sample
    train_file = parse_args().train_sample
    model_dir = parse_args().model_result_dir


    mean_data = load_mean_data(file_name=mean_file)
    val_x , val_y = load_val_data(mean_data , file_name=val_file)

    # 模型定义。-----------------------------------
    feature_len = len(val_x[0])
    print(feature_len)

    input = layers.Input(shape=(feature_len,), name="input")
    temp = FM(1)(input)
    output = Activation(K.sigmoid)(temp)

    model = models.Model([input], output)

    model.compile(
        # optimizer=optimizers.Adagrad(lr=0.001),
        optimizer = optimizers.sgd(lr=0.001),
        loss=losses.binary_crossentropy,
        metrics=[metrics.binary_accuracy])
    model.summary()


    # 训练过程。-------------------------
    EPOCH_NUM = 100
    LOG_INFO_STEP = 100
    for epoch in range(EPOCH_NUM):
        metrics_list = model.metrics_names
        step = 0
        for batch_x , batch_y in train_data_generator(mean_data , file_name=train_file , batch_size=512):
            # print("batch_x: " + str(batch_x.shape))
            # print("batch_y: " + str(batch_y.shape))
            # print()
            step = step + 1
            train_cost = model.train_on_batch(batch_x, batch_y)


            if step % LOG_INFO_STEP == 0:
                train_log = []
                for metric , train_v   in zip(metrics_list , train_cost ):
                    train_log += [ "train_%s : %.6f " % ( metric , train_v ) ]
                print("epoch:%d step:%d ;%s " % (epoch , step, ",".join(train_log) ))

        # 每个 epoch 输出一次val信息
        val_cost = model.evaluate(val_x, val_y)  # 评估网络 [loss , acc]
        val_log = []
        for metric  , val_v   in zip(metrics_list , val_cost):
            val_log += [ "val_%s : %.6f " % ( metric , val_v ) ]
        print("endEpoch:%d ;%s" % (epoch ,  ",".join(val_log)))
        model.save(model_dir + "/endEpoch"+str(epoch)+".h5")

    # 训练过程。-------------------------
    # model.fit_generator(train_data_generator(mean_data) , validation_data=(val_x , val_y) , steps_per_epoch=2870 , epochs=EPOCH_NUM)


    # plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True) # plot my model

    # model.fit({"input":train_x}, train_y,
    #           verbose=1, epochs=100, batch_size=256,
    #           validation_data = ( {"input":val_x} , val_y ))

