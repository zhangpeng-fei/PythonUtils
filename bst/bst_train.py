import os
os.environ['KMP_WARNINGS'] = 'off'
import sys
if "win" in sys.platform :
    from .bst_attention import *
    from .bst_data_loader import *
    from .bst_callbacks import *
else :
    from bst_attention import *
    from bst_data_loader import *
    from bst_callbacks import *
#from .selfAttention import Self_Attention
from keras import models , layers , optimizers ,losses , metrics, backend
import numpy as np
import keras
from  keras.layers import  *
from keras.models import *
from keras.utils import to_categorical

import json
import numpy as np
from random import  shuffle

"""
@description：bst模型训练过程。
"""

# read profile data  读取画像数据
os.system("echo \" +++++start load_profile_data `date '+%Y-%m-%d %H:%M:%S'` \"")
dic_feature_idx , dic_profile_vid = load_profile_data(save_feature=True)
os.system("echo \" +++++finish load_profile_data `date '+%Y-%m-%d %H:%M:%S'` \"")
# read train sample data  读取训练样本数据。
os.system("echo \" +++++start load_train_data `date '+%Y-%m-%d %H:%M:%S'` \"")
feature_vid_year , \
data_vid , data_posid,data_cid1,data_cid2,data_cid3,data_cid4, \
labels = load_train_data(dic_profile_vid)
os.system("echo \" +++++finish load_train_data `date '+%Y-%m-%d %H:%M:%S'` \"")

# 划分训练集和验证集----------------------------------------
vid_size = len(dic_profile_vid)
posid_size = np.max(data_posid) + 1
cate_size = len(dic_feature_idx["vid_cates"])
year_size = len(dic_feature_idx["vid_year"])
print("vid_size :" + str(vid_size))
print("posid_size :" + str(posid_size))
print("cate_size :" + str(cate_size))
print("year_size :" + str(year_size))


train_feature_vid_year = feature_vid_year[:-90000]
train_data_vid = data_vid[:-90000]
train_data_posid = data_posid[:-90000]
train_data_cid1 = data_cid1[:-90000]
train_data_cid2 = data_cid2[:-90000]
train_data_cid3 = data_cid3[:-90000]
train_data_cid4 = data_cid4[:-90000]
train_labels = labels[:-90000]

val_feature_vid_year = feature_vid_year[-90000:]
val_data_vid = data_vid[-90000:]
val_data_posid = data_posid[-90000:]
val_data_cid1 = data_cid1[-90000:]
val_data_cid2 = data_cid2[-90000:]
val_data_cid3 = data_cid3[-90000:]
val_data_cid4 = data_cid4[-90000:]
val_labels = labels[-90000:]

seq_size = 20            # 序列长度
emb_dim = 32

# model structure 模型定义和训练 -------------------------------------------
vid_input = Input(shape=(seq_size,), dtype='int32',name="vid_input")
vid_emb = Embedding(vid_size,emb_dim)(vid_input)

posid_input = Input(shape=(seq_size,),dtype="int32",name="posid_input")
posid_emb = Embedding(posid_size,emb_dim)(posid_input)

cid_input_1 = Input(shape=(seq_size,),dtype="int32",name="cid_input_1")
cid_input_2 = Input(shape=(seq_size,),dtype="int32",name="cid_input_2")
cid_input_3 = Input(shape=(seq_size,),dtype="int32",name="cid_input_3")
cid_input_4 = Input(shape=(seq_size,),dtype="int32",name="cid_input_4")
cid_emb = Embedding(cate_size,emb_dim)
cid_emb_1 = cid_emb(cid_input_1)
cid_emb_2 = cid_emb(cid_input_2)
cid_emb_3 = cid_emb(cid_input_3)
cid_emb_4 = cid_emb(cid_input_4)

feature_year_input = Input(shape=(1,),dtype="int32",name="year_input")
time_emb = Embedding(year_size,emb_dim)(feature_year_input)
time_flattern = Flatten()(time_emb)
#-----------------------------------------------------
att_concat = layers.concatenate([vid_emb  , posid_emb ,cid_emb_1 , cid_emb_2,cid_emb_3,cid_emb_4 ],axis=-1)
att_emb  = Multy_Head_Attention(inner_dim=64 , head_num=8)(att_concat)
att_flattern = Flatten()(att_emb)

dnn_concat = layers.concatenate([att_flattern ,time_flattern ],axis=-1)


temp = Dense(1024, activation='relu')(dnn_concat)
temp = Dense(512, activation='relu')(temp)
temp = Dense(256, activation='relu')(temp) #
temp = Dense(128, activation='relu')(temp)
output = Dense(1, activation='sigmoid')(temp)

model = Model([vid_input , posid_input ,cid_input_1 ,cid_input_2,cid_input_3,cid_input_4,feature_year_input], output)
model.compile(optimizer=optimizers.Adagrad(lr=0.001),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])
#  bst论文中 ， 使用logloss作为损失函数，Aagrad优化器
model.summary()

model.fit({"vid_input": train_data_vid,
           "posid_input": train_data_posid,
           "cid_input_1": train_data_cid1,
           "cid_input_2": train_data_cid2,
           "cid_input_3": train_data_cid3,
           "cid_input_4": train_data_cid4,
           "year_input": train_feature_vid_year},
          train_labels,verbose=2, epochs=50, batch_size=256,

          callbacks = [roc_callback(
              training_data=[[train_data_vid,train_data_posid,train_data_cid1,train_data_cid2,train_data_cid3,train_data_cid4,train_feature_vid_year],train_labels ],
              validation_data=[[val_data_vid,val_data_posid,val_data_cid1,val_data_cid2,val_data_cid3,val_data_cid4,val_feature_vid_year],val_labels])] ,

          validation_data=({"vid_input": val_data_vid,
                            "posid_input": val_data_posid,
                            "cid_input_1": val_data_cid1,
                            "cid_input_2": val_data_cid2,
                            "cid_input_3": val_data_cid3,
                            "cid_input_4": val_data_cid4,
                            "year_input": val_feature_vid_year},
                           val_labels))

# model.fit(X_train, y_train,
#           epochs=10, batch_size=4,
#           callbacks = [roc_callback(training_data=[X_train, y_train], validation_data=[X_test, y_test])] )
#
#
# plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True) # plot my model
