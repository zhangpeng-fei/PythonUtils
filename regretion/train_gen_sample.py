
import numpy as np
from random import  shuffle
"""
@description：生成lR原始训练样本。

15天总次数；白天总次数；晚上总次数； |
平均每天次数；平均白天次数；平均晚上次数。
1-15天，每天次数，每天白天次数，每天晚上次数。

15天总次数，白天总次数，晚上总次数；15天总时长，白天总时长，晚上总时长；
平均每天次数，平均白天次数，平均晚上次数；平均每天时长，平均白天时长，平均晚上时长
每天次数，每天白天次数，每天晚上次数；每天时长，每天白天时长，每天晚上时长。

// dnum : (15+2) * 3
// dnum : (15+2) * 6
"""




# 读dnum-info文件
def load_dnum_flag():
    # dnum : 0|1
    dic_dnum_flag = dict()  # {dnum : 0|1}

    for line in open("dnum-info"):
        line_list = line.strip().split(":")
        if len(line_list) != 2:
            continue
        dnum = line_list[0].strip()
        flag = line_list[1].strip()
        if dnum not in dic_dnum_flag:
            dic_dnum_flag[dnum] = int(flag)

    print("dnum_len:" + str(len(dic_dnum_flag)))
    return dic_dnum_flag

def load_sample_info(dic_dnum_flag):
    #     dnum: (15 + 2) * 3 = 51
    #     dnum: (15 + 2) * 6 = 102 view
    #     dnum:
    #     dnum + ":" + leftStr + "," + rightStr

    dic_channel_behav_info = dict(dict())
    # {channel : { behav : { dnum : info } }}

    channel_list = ["child", "comic", "movie", "tv", "variety"]
    behav_list = ["collect", "pay", "search", "view"]

    for channel in channel_list:
        dic_behav_info = dict()
        for behav in behav_list:

            dic_info = dict()
            for line in open(channel + "-" + behav, "r"):
                line_list = line.strip().split(":")
                if len(line_list) != 2:
                    continue
                dnum = line_list[0].strip()
                info_list = line_list[1].strip().split(",")
                if dnum not in dic_dnum_flag:
                    continue
                if behav == "view" and len(info_list) != 102:
                    continue
                if behav != "view" and len(info_list) != 51:
                    continue
                if dnum not in dic_info:
                    dic_info[dnum] = [int(info) for info in info_list]
            print("readDone " + channel +"-" + behav + " "+str(len(dic_info)))

            dic_behav_info[behav] = dic_info

        dic_channel_behav_info[channel] = dic_behav_info

    # 生成矩阵，样本。

    data_x = []
    data_y = []
    print("-------------------")
    for channel,behav_info in dic_channel_behav_info.items():
        print(channel + " : " + str(len(behav_info)))
        for behav , info in behav_info.items():
            print(channel+"-"+behav+":"+str(len(info))  )
    print("-------------------")



    for dnum in dic_dnum_flag:
        dnum_total_info = []
        for channel in channel_list:
            for behav in behav_list:
                inner_dnum_info_dic = dic_channel_behav_info[channel][behav]
                if dnum in inner_dnum_info_dic:
                    dnum_total_info = dnum_total_info + inner_dnum_info_dic[dnum]
                elif behav == "view" :
                    dnum_total_info = dnum_total_info + [ 0 for i in range(102)]
                else:
                    dnum_total_info = dnum_total_info + [0 for i in range(51)]

        data_x.append(dnum_total_info)
        data_y.append([dic_dnum_flag[dnum]])

    data_x = np.array(data_x, dtype="int32")
    data_y = np.array(data_y, dtype="int32")
    return data_x , data_y



dic_dnum_flag = load_dnum_flag()

data_x , data_y = load_sample_info(dic_dnum_flag)

"""
input       [b,len]     int32          0|1
labels      [b,1]       float32        0.0|1.0
"""
# shuffle

idx = [i for i in range(len(data_x))]
shuffle(idx)
data_x = data_x[idx, :]
data_y = data_y[idx, :]


# 写训练样
writer = open("sample_train","w")
for x , y in zip(data_x , data_y):
    x = [  str(i)  for i in x ]
    y = [  str(i)   for i in y ]
    writer.write(",".join(x + y) + "\n")
    writer.flush()
writer.flush() ; writer.close()




# feature_len = 100
#
# feature_input = layers.Input(shape=(feature_len,),dtype="int32",name="feature_input")
# output  = layers.Dense(1,activation='sigmoid')(feature_input)
#
# model = models.Model([feature_input], output)
#
# model.compile(optimizer=optimizers.Adagrad(lr=0.001),
#               loss=losses.binary_crossentropy,
#               metrics=[metrics.binary_accuracy])
#
# model.fit(data_x, data_y,
#           verbose=1 , epochs=10, batch_size=128,
#           validation_data = ( val_x , val_y ))

