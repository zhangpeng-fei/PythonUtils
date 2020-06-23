
import numpy as np
"""
@description：降维后的数据读取方法定义。
"""

# @description：读取所有的样本数据，并做离散化。返回ndarry
def one_hot_load_sample_data(mean_file , sample_file):

    # load mean data
    mean_data = []
    for line in open(mean_file): # 一行。
        num_list = line.strip().split(",")
        num_list = [ float(i.strip()) for i in num_list]
        mean_data.append(num_list)
    mean_data = mean_data[0]
    mean_data = np.array(mean_data , dtype="float32")

    #  load sample data
    x = []
    y = []
    for line in open(sample_file):
        line_list = line.strip().split(":")
        if len(line_list) != 2 :
            continue
        temp_x = line_list[0].strip().split(",")
        temp_x = [ float(i.strip()) for i in temp_x ]
        temp_x = np.array(temp_x  , dtype="float32")
        temp_x = temp_x > mean_data
        result_x = []
        for i in temp_x:
            if i is True:
                result_x = result_x + [1.0 , 0.0]
            else :
                result_x = result_x + [0.0 , 1.0]
        result_y = [float(line_list[1].strip())]

        x.append(result_x)
        y.append(result_y)
    x = np.array(x, dtype="float32")
    y = np.array(y, dtype="float32")
    print("x.shape : %s" % str(x.shape))
    print("y.shape : %s" % str(y.shape))
    return x, y

# @description：读取所有的样本数据，不做离散化。返回ndarry
def load_sample_data(file_name):
    # 78:1
    x = []
    y = []

    for line in open(file_name,"r"):
        line_list = line.strip().split(":")
        if len(line_list) != 2 :
            continue
        temp_x = line_list[0].strip().split(",")
        temp_x = [ float(i.strip()) for i  in temp_x ]
        temp_y = [ float(line_list[1].strip()) ]
        x.append(temp_x)
        y.append(temp_y)
    x = np.array(x , dtype="float32")
    y = np.array(y , dtype="float32")
    print("x.shape : %s" % str(x.shape))
    print("y.shape : %s" % str(y.shape))
    return x , y

# @description：训练样本生成器，返回结果为离散化后的ndarry
def one_hot_generator(mean_file , sample_file , batch_size = 256):
    # load mean data
    mean_data = []
    for line in open(mean_file): # 一行
        num_list = line.strip().split(",")
        num_list = [float(i.strip()) for i in num_list]
        mean_data.append(num_list)
    mean_data = mean_data[0]
    mean_data = np.array(mean_data , dtype="float32")

    count = 0
    batch_x = []
    batch_y = []
    for line in open(sample_file):

        line_list = line.strip().split(":")
        if len(line_list) !=2:
            continue

        temp_x = line_list[0].strip().split(",")
        temp_x = [ float(i.strip()) for i in temp_x ]
        temp_x = np.array(temp_x , dtype="float32")
        temp_x = temp_x > mean_data
        result_x = []
        for i in temp_x:
            if i is True :
                result_x = result_x + [1.0 , 0.0]
            else :
                result_x = result_x + [0.0 , 1.0]
        result_y = [ float(line_list[1].strip()) ]

        batch_x.append(result_x)
        batch_y.append(result_y)
        count = count + 1
        if count  == batch_size:
            batch_x = np.array(batch_x , dtype="float32")
            batch_y = np.array(batch_y , dtype="float32")
            yield  batch_x , batch_y
            count = 0
            batch_x = []
            batch_y = []

# @description：训练样本生成器，返回结果为离散化前的ndarry
def data_generator(file_name , batch_size = 256):

    batch_x = []
    batch_y = []

    count = 0
    for line in open(file_name , "r"):
        line_list = line.strip().split(":")
        if len(line_list) !=2 :
            continue
        temp_x = line_list[0].strip().split(",")
        temp_x = [ float(v.strip()) for v in temp_x ]
        temp_y = [ float(line_list[1].strip()) ]

        batch_x.append(temp_x)
        batch_y.append(temp_y)
        count  += 1
        if(count == batch_size):
            batch_x = np.array(batch_x , dtype="float32")
            batch_y = np.array(batch_y , dtype="float32")

            yield batch_x , batch_y
            count = 0
            batch_x = []
            batch_y = []
