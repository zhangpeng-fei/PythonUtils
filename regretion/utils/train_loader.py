import numpy as np
"""
@description：降维前的样本数据读取方法的定义
@author：zhangpengfei
@date:2020/04/29
"""


# @description：读取验证集数据为ndarry
# @return : val_x , val_y
def load_val_data( mean_data , file_name="data_train/val_sample" ):

    val_x = []
    val_y = []
    for line in open(file_name):
        # print(line)

        x_list = []
        y_list = []

        line_list = line.strip().split(",")
        # print(len(line_list))
        if len(line_list)  != 1276   :
            continue
        num_list = [ float(i.strip())  for i in line_list ]

        temp_x_list = np.array(num_list[:-1] , dtype="float32") # array([])
        temp_x_list = temp_x_list > mean_data
        temp_x_list = np.array(temp_x_list , dtype="int32")

        for x in temp_x_list:
            if x==0:
                x_list += [1.0,0.0]
            else:
                x_list += [0.0,1.0]
        y_list += num_list[-1:]

        val_x.append(x_list)
        val_y.append(y_list)

    val_x = np.array(val_x,dtype="float32")
    val_y = np.array(val_y,dtype="float32")
    print("val_x: "+str(val_x.shape))
    print("val_y: "+str(val_y.shape))
    return val_x , val_y

# @description：训练集数据生成器
# @return ： batch_x , batch_y
def train_data_generator(mean_data ,  file_name="data_train/train_sample" , batch_size = 512):

    batch_x = []
    batch_y = []
    i = 0
    # with open(file_name,"r") as f:
    for line in open(file_name,"r"):

        x_list = []
        y_list = []

        line_list = line.strip().split(",")
        if len(line_list) !=1276:
            continue
        num_list = [ float(i.strip()) for i in line_list]

        temp_x_list = np.array(num_list[:-1] , dtype="float32" )
        temp_x_list = temp_x_list > mean_data
        temp_x_list = np.array(temp_x_list , dtype="int32")

        for x in temp_x_list:
            if x==0 :
                x_list += [1.0,0.0]
            else:
                x_list += [0.0,1.0]
        y_list += num_list[-1:]

        batch_x.append(x_list)
        batch_y.append(y_list)
        i = i + 1

        if(i == batch_size):
            batch_x = np.array(batch_x , dtype="float32")
            batch_y = np.array(batch_y, dtype="float32")
            # print("batch_x: " + str(batch_x.shape))
            # print("batch_y: " + str(batch_y.shape))
            # print(num_list[:5])
            # print(batch_x[0][140:160])
            yield batch_x, batch_y
            i = 0
            batch_x = []
            batch_y = []
