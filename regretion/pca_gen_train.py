
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import numpy as np
import argparse

"""
@description：使用pca对原始训练样本降维，生成新的训练样本。

1、对于所有特征的训练样本：
    读取为numpy。
    标准化 -》 保存标准化模型数据。
    PCA降维度。保存模型数据。
    训练。保存模型数据。
2、infer
    标准化。
    PCA降低维度。
    预测。
    
keraspython -u pca_sample_train.py  \
--source_train_sample_file=data_train/train_source/source_sample   \
--save_stand_model_file=data_both/standard.model  \
--save_pca_model_file=data_both/pca.model  \
--save_train_sample_file=data_train/train_sample/sample_total  \
>  log_pca_sample_train  2>&1  & 

我要使用这几个方法：
1、标准化+pca ->  lr
2、标准化+pca + 平均值离散化 -> lr 
"""

def parse_args():
    parser = argparse.ArgumentParser(description="  Gen pca model and train sample ")
    parser.add_argument('--source_train_sample_file', type=str, default='',
                        help='Input  source train sample file ')
    parser.add_argument('--save_stand_model_file', type=str, default='',
                        help='Input  save standard model file ')
    parser.add_argument('--save_pca_model_file', type=str, default='',
                        help='Input  save pca model file ')
    parser.add_argument('--save_train_sample_file', type=str, default='',
                        help='Input  save train sample file ')
    return parser.parse_args()

if __name__ == "__main__":
    source_train_sample_file = parse_args().source_train_sample_file
    save_stand_model_file = parse_args().save_stand_model_file
    save_pca_model_file = parse_args().save_pca_model_file
    save_train_sample_file = parse_args().save_train_sample_file

    # 读取原始的训练数据文件。
    train_x = []
    train_y = []

    for line in open(source_train_sample_file , "r"):
        line_list = line.strip().split(",")
        if len(line_list) != 1276 :
            continue
        num_list = [ float(i.strip()) for i in line_list ]
        x = num_list[:-1]
        y = num_list[-1:]
        train_x.append(x)
        train_y.append(y)

    train_x = np.array(train_x , dtype="float32")
    train_y = np.array(train_y , dtype="float32")
    print("source train_x.shape:%s" % str(train_x.shape))
    print("source train_y.shape:%s" % str(train_y.shape))

    # 先做标准化
    standard_model = StandardScaler() # 方差为1，均值为0。
    train_x = standard_model.fit_transform(train_x)
    joblib.dump(standard_model, save_stand_model_file)
    print("stand  train_x.shape:%s" % str(train_x.shape))
    print("source train_y.shape:%s" % str(train_y.shape))

    # 训练 pca 模型，并保存pca模型
    pca_model = PCA(n_components=0.999)
    train_x = pca_model.fit_transform(train_x)
    joblib.dump(pca_model, save_pca_model_file)
    print("pca    train_x.shape:%s" % str(train_x.shape))
    print("source train_y.shape:%s" % str(train_y.shape))

    # print(train_x)
    # print(train_y)
    # 保存新的训练样本
    writer = open(save_train_sample_file,"w")
    for x , y in zip( train_x , train_y ):
        x_str = ",".join([ str(i) for i in x])
        y_str = str(y[0])
        writer.write(x_str + ":" + y_str + "\n")
        writer.flush()
    writer.flush()
    writer.close()




