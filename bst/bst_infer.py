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
from keras.models import *

"""
@description：bst模型infer过程。
"""

os.system("echo \" start infer `date '+%Y-%m-%d %H:%M:%S'` \"")
model = load_model('movie_BST_1.h5', custom_objects={'Multy_Head_Attention': Multy_Head_Attention})
model.summary()

# read profile data
dic_feature_idx , dic_profile_vid = load_profile_data()
# read infer vids
infer_vid_list = load_infer_vids(dic_profile_vid)
batch_size = len(infer_vid_list)

seq_size = 20 # 总的序列长度。
topk = 30
writer = open("result-infer","w")
# dnum:vid_#_posid,vid_#_posid,...   这里 infer 样本中保证 dnum 是唯一的。

for line in open("vid-infer","r"):
    feature_vid_year = []

    data_vid = []
    data_posid = []
    data_cid1 = []
    data_cid2 = []
    data_cid3 = []
    data_cid4 = []
    #data_year = []
    #----------------
    vid_list = []
    posid_list = []
    cid1_list = []
    cid2_list = []
    cid3_list = []
    cid4_list = []
    year_list = []

    line_list = line.split(":")
    if len(line_list) != 2 :
        continue
    dnum = line_list[0].strip()
    infoStr = line_list[1].strip()
    # 过滤 vid 不存在的info  。
    source_vid_posid_list = infoStr.split(",")
    legal_vid_posid_list = [ info for info in source_vid_posid_list \
                 if info.split("_#_")[0].strip() in dic_profile_vid ]
    if len(legal_vid_posid_list) == 0 :
        continue
    # 补齐 序列长度。
    legal_vid_posid_list = legal_vid_posid_list + legal_vid_posid_list[-1:] * seq_size
    legal_vid_posid_list = legal_vid_posid_list[:seq_size - 1] # 现在是19个。

    for vid_pid in legal_vid_posid_list :
        vid = vid_pid.split("_#_")[0].strip()
        pid = vid_pid.split("_#_")[1].strip()
        # dic_feature_idx , dic_profile_vid = load_profile_data()
        vid_list.append(dic_profile_vid[vid]["idx"])
        posid_list.append(int(pid))
        cid1_list.append(dic_profile_vid[vid]["cates"][0])
        cid2_list.append(dic_profile_vid[vid]["cates"][1])
        cid3_list.append(dic_profile_vid[vid]["cates"][2])
        cid4_list.append(dic_profile_vid[vid]["cates"][3])
        year_list.append(dic_profile_vid[vid]["year"])

    for infer_vid in infer_vid_list:
        feature_vid_year.append([dic_profile_vid[infer_vid]["year"]])

        data_vid.append(vid_list + [dic_profile_vid[infer_vid]["idx"]] )
        data_posid.append(posid_list + [0])
        data_cid1.append(cid1_list + [dic_profile_vid[infer_vid]["cates"][0]])
        data_cid2.append(cid2_list + [dic_profile_vid[infer_vid]["cates"][1]])
        data_cid3.append(cid3_list + [dic_profile_vid[infer_vid]["cates"][2]])
        data_cid4.append(cid4_list + [dic_profile_vid[infer_vid]["cates"][3]])

    feature_vid_year = np.array(feature_vid_year, dtype="int32")
    data_vid = np.array(data_vid, dtype="int32")
    data_posid = np.array(data_posid, dtype="int32")
    data_cid1 = np.array(data_cid1, dtype="int32")
    data_cid2 = np.array(data_cid2, dtype="int32")
    data_cid3 = np.array(data_cid3, dtype="int32")
    data_cid4 = np.array(data_cid4, dtype="int32")


    # model = Model([vid_input , posid_input ,cid_input_1 ,cid_input_2,cid_input_3,cid_input_4,feature_year_input], output)
    predict_result = model.predict([data_vid,data_posid, data_cid1 ,data_cid2 , data_cid3,data_cid4,feature_vid_year]) # [b,1]
    predict_score = np.reshape(predict_result,batch_size) # [b,]  TODO
    #print(predict_score)
    # arr = np.array([1, 3, 2, 4, 5])
    top_k_idx = predict_score.argsort()[::-1][0:topk]  # TODO
    #print(top_k_idx)
    # dnum:vid_#_score,vid_#_score
    writer.write(dnum + ":")
    for idx in top_k_idx :
        writer.write(infer_vid_list[idx]+"_#_{:.6}".format(predict_score[idx]))
        if idx != top_k_idx[-1] :
            writer.write(",")
    writer.write("\n")
    writer.flush()

writer.flush() ; writer.close()
os.system("echo \" finish infer `date '+%Y-%m-%d %H:%M:%S'` \"")



























