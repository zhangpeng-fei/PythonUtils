import  numpy as np
import argparse
from random import  shuffle

"""
@description：生成训练样本，各个特征的平均值。为原始样本的离散化做准备。

keraspython -u both_gen_mean_data.py   \
--train_sample=data_train/train_sample   \
--infer_sample=data_infer/infer_sample   \
--save_mean_data=data_both/mean_data   \
>   log   2>&1  &    
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Gen mean data")
    parser.add_argument('--train_sample', type=str, default='data_train/train_sample',
                        help='Input train sample ')
    parser.add_argument('--infer_sample', type=str, default='data_infer/infer_sample',
                        help='Input infer sample')
    parser.add_argument('--save_mean_data', type=str, default='data_both/mean_data',
                        help='Input save mean file')
    return parser.parse_args()

if __name__ == "__main__":
    train_sample = parse_args().train_sample
    infer_sample = parse_args().infer_sample
    save_mean_file = parse_args().save_mean_file

    # 读一行数据
    file = open(infer_sample,"r") # 0,0,0..0,0,1:dnum
    mean_data = file.readline().strip().split(":")[0].strip().split(",")
    mean_data = [ float(i.strip()) for i in mean_data ]
    mean_data = np.array([mean_data] , dtype="float32") # array([1,15575] ,"float32")
    file.close()
    print("mean_data : " + str(mean_data.shape))
    print("------------------")

    # infer 样本的平均值
    for line in open(infer_sample,"r"):
        source_line_list = line.strip().split(":")
        if len(source_line_list) != 2 :
            continue
        line_list = source_line_list[0].strip().split(",")
        if len(line_list) != 1275 :
            continue
        nums = [ float(i.strip()) for i in line_list ]
        nums = np.array([nums] , dtype="float32")
        # print("nums : " + str(nums.shape))            #  [1,1275]
        mean_data = np.concatenate((mean_data , nums ) , axis=0)
        # print("mean_data : " + str(mean_data.shape)) #  [2,1275]
        mean_data = np.mean(mean_data , axis=0)
        # print("mean_data : " + str(mean_data.shape)) #  [1275,]
        mean_data = np.array([mean_data] , dtype="float32")
        # print("mean_data : " + str(mean_data.shape)) #  [1,1275]

    # train 样本的平均值
    for line in open(train_sample,"r"): # 0,0,0,0,1,1,1,....
        line_list = line.strip().split(",")
        if len(line_list) !=1275 :
            continue
        nums = [ float(i.strip()) for i in line_list ]
        nums = nums[:-1]
        nums = np.array([nums] , dtype="float32")
        mean_data = np.concatenate(( mean_data , nums ) , axis=0)
        mean_data = np.mean( mean_data , axis=0 )
        mean_data = np.array([mean_data] , dtype="float32")



    # 写样本的平均值。
    writer = open(save_mean_file,"w")
    writer.write(",".join([ str(i) for i in mean_data[0] ]))
    writer.flush()
    writer.close()

