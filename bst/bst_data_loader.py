
import json
import numpy as np
from random import  shuffle
"""
@description：定义数据读取的方法。
"""

# 读取数据。
def load_profile_data(save_feature = True):

    feature_list = ["vid_cates","vid_year"]
    dic_feature_idx = dict() # {vid_cates:dic_cate_idx , vid_year:dic_year_idx , ...}
    for feature in feature_list:
        dic_temp = dict()
        dic_temp["default"] = 0
        dic_feature_idx[feature] = dic_temp

    # vid画像：补齐格式 + 完成转换。
    dic_profile_vid = dict() # {vid : { year:year_info , cate:cate_info , feature:info , .... } }
    for jsonStr in open("vid-profile", "r"):
        # jsonStr =  {"vid":"10729808000","year":"1xxx","cates":"当代,普通话,华语,片库"}
        line_dic = json.loads(jsonStr)
        temp_dic = dict()
        vid = line_dic["vid"]
        if vid in dic_profile_vid :
            continue

        temp_dic["idx"] = len(dic_profile_vid)

        year = line_dic["year"]
        if year not in dic_feature_idx["vid_year"]:
            dic_feature_idx["vid_year"][year] = len(dic_feature_idx["vid_year"])
        year_idx = dic_feature_idx["vid_year"][year]
        temp_dic["year"]=year_idx

        temp_list = line_dic["cates"].strip().split(",") + ["default","default","default","default"]
        cate_list = [i for i in temp_list if len(i) > 0]
        cate_idx_list = []
        for i in range(4):
            cate = cate_list[i]
            if cate not in dic_feature_idx["vid_cates"]:
                dic_feature_idx["vid_cates"][cate] = len(dic_feature_idx["vid_cates"])
            cate_idx_list.append(dic_feature_idx["vid_cates"][cate])
        temp_dic["cates"]=cate_idx_list

        dic_profile_vid[vid] = temp_dic
    # dnum画像： 补齐格式 + 完成转换。
    # dic_profile_dnum = dict()
    # for jsonStr in open("profile-dnum", "r"):
    if save_feature:
        save_feature_idx(dic_feature_idx)
    return dic_feature_idx , dic_profile_vid

def save_feature_idx(dic_feature_idx):
    for feature , feature_idx in dic_feature_idx.items():
        writer = open(feature, "w")
        for name , idx in feature_idx.items():
            writer.write(name + ":" + str(idx) + "\n")
        writer.flush()
        writer.close()

# 读取训练数据。
def load_train_data(dic_profile_vid):
    feature_vid_year = []

    data_vid = []
    data_posid = []
    data_cid1 = []
    data_cid2 = []
    data_cid3 = []
    data_cid4 = []
    #data_year = []

    labels = []
    # 523778415:aggiy0qv89a47k3_#_3,zr5a67l333ehzu9_#_2:C
    for line in open("sample", "r"):
        line_list = line.strip().split(":")
        if len(line_list) != 3 :
            continue
        dnum = line_list[0].strip()
        seqInfo = line_list[1].strip()
        flag = line_list[2].strip()

        # 剔除 vid 不符合条件的样本。
        vid_posid_list = seqInfo.split(",") # [vid_#_posid,...]
        vid_list = [ vid_posid.split("_#_")[0].strip()  for vid_posid in  vid_posid_list \
                     if  vid_posid.split("_#_")[0].strip() in  dic_profile_vid  ]

        if len(vid_list)  != 20 :
            continue

        vid_list = []
        posid_list = []
        cid1_list = []
        cid2_list = []
        cid3_list = []
        cid4_list = []
        year_list = []
        # print("\n"+line)
        for vid_posid in vid_posid_list:
            vid = vid_posid.split("_#_")[0].strip()
            posid = vid_posid.split("_#_")[1].strip()

            vid_list.append(dic_profile_vid[vid]["idx"])
            posid_list.append(int(posid))
            cid1_list.append(dic_profile_vid[vid]["cates"][0])
            cid2_list.append(dic_profile_vid[vid]["cates"][1])
            cid3_list.append(dic_profile_vid[vid]["cates"][2])
            cid4_list.append(dic_profile_vid[vid]["cates"][3])
            year_list.append(dic_profile_vid[vid]["year"])

        feature_vid_year.append(year_list[-1])
        data_vid.append(vid_list)   # 序列数据。
        data_posid.append(posid_list)
        data_cid1.append(cid1_list)
        data_cid2.append(cid2_list)
        data_cid3.append(cid3_list)
        data_cid4.append(cid4_list)
        #data_year.append(year_list)
        if flag == "C" :
            labels.append(1.)
        else :
            labels.append(0.)
    # 转 Numpy 数组
    feature_vid_year = np.reshape(np.asarray(feature_vid_year, dtype="int32"), (len(feature_vid_year), 1))

    data_vid = np.array(data_vid, dtype="int32")
    data_posid = np.array(data_posid, dtype="int32")
    data_cid1 = np.array(data_cid1, dtype="int32")
    data_cid2 = np.array(data_cid2, dtype="int32")
    data_cid3 = np.array(data_cid3, dtype="int32")
    data_cid4 = np.array(data_cid4, dtype="int32")
    #data_year = np.array(data_year, dtype="int32")

    labels = np.reshape(np.asarray(labels, dtype="float32"), (len(labels), 1))

    print("feature_vid_year-----------")
    print(feature_vid_year[1])
    print(feature_vid_year.shape)
    print(feature_vid_year.dtype)
    print(np.max(feature_vid_year))
    print(np.min(feature_vid_year))
    print("data_vid-----------")
    print(data_vid[1])
    print(data_vid.shape)
    print(data_vid.dtype)
    print(np.max(data_vid))
    print(np.min(data_vid))
    print("data_posid-----------")
    print(data_posid[1])
    print(data_posid.shape)
    print(data_posid.dtype)
    print(np.max(data_posid))
    print(np.min(data_posid))
    print("data_cid1-----------")
    print(data_cid1[1])
    print(data_cid1.shape)
    print(data_cid1.dtype)
    print(np.max(data_cid1))
    print(np.min(data_cid1))
    print("data_cid2-----------")
    print(data_cid2[1])
    print(data_cid2.shape)
    print(data_cid2.dtype)
    print(np.max(data_cid2))
    print(np.min(data_cid2))
    print("data_cid3-----------")
    print(data_cid3[1])
    print(data_cid3.shape)
    print(data_cid3.dtype)
    print(np.max(data_cid3))
    print(np.min(data_cid3))
    print("data_cid4-----------")
    print(data_cid4[1])
    print(data_cid4.shape)
    print(data_cid4.dtype)
    print(np.max(data_cid4))
    print(np.min(data_cid4))
    # print("data_year-----------")
    # print(data_year[1])
    # print(data_year.shape)
    # print(data_year.dtype)
    # print(np.max(data_year))
    # print(np.min(data_year))
    print("labels-----------")
    print(labels[1])
    print(labels.shape)
    print(labels.dtype)
    print(np.max(labels))
    print(np.min(labels))
    print("==========================")

    idx = [i for i in range(len(data_vid))]
    shuffle(idx)
    feature_vid_year = feature_vid_year[idx, :]
    data_vid = data_vid[idx, :]
    data_posid = data_posid[idx, :]
    data_cid1 = data_cid1[idx, :]
    data_cid2 = data_cid2[idx, :]
    data_cid3 = data_cid3[idx, :]
    data_cid4 = data_cid4[idx, :]
    #data_year = data_year[idx, :]
    labels = labels[idx, :]
    # timeID_data = timeID_data[idx, :][:db_size]
    return feature_vid_year , \
           data_vid , data_posid,data_cid1,data_cid2,data_cid3,data_cid4,  \
           labels

# load  infer  vids .
def load_infer_vids(dic_profile_vid):
    infer_vid_list = []
    for line in open("infer-vid"):
        vid = line.strip()
        if vid in dic_profile_vid  and  vid not in infer_vid_list :
            infer_vid_list.append(vid)
    return infer_vid_list


