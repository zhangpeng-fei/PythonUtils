import numpy as np

"""
@description：定义通用的数据读取方法
"""
# @description：读取样本特征平均值数据为 ndarry
# @return = ndarray([1275,] , "float32")
def load_mean_data(file_name="data_both/mean_data"):
    mean_data = []
    for line in open(file_name , "r"):
        line_list = line.strip().split(",")
        if len(line_list) != 1275:
            continue
        num_list = [ float(i.strip())  for i in line_list ]
        mean_data.append(num_list)
    mean_data = np.array(mean_data,dtype='float32')[0]
    print("mean_data: " + str(mean_data.shape))
    return mean_data # ndarray([1275,] , "float32")
