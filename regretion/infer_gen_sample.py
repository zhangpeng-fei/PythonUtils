

import numpy as np

"""
@description：生成原始的infer样本：

1、需要把所有的东西读进去。 读成一个dict ； 一共 4.5G .
    同时取出所有的dnum.
2、生成所有的 infer_sample ：    序列长度：dnum
"""

def gen_sample_infer():
    #     dnum: (15 + 2) * 3 = 51
    #     dnum: (15 + 2) * 6 = 102 view
    #     dnum:
    #     dnum + ":" + leftStr + "," + rightStr

    set_dnum = set()

    dic_channel_behav_info = dict(dict())
    # {channel : { behav : { dnum : info } }}

    channel_list = ["child", "comic", "movie", "tv", "variety"]
    behav_list = ["collect", "pay", "search", "view"]

    for channel in channel_list:
        dic_behav_info = dict()
        for behav in behav_list:

            dic_info = dict()
            temp_set_dnum = set()
            for line in open(channel + "-" + behav, "r"):
                #line  =  dnum:0,2,0,0,0,1,1,1,0...
                line_list = line.strip().split(":")
                if len(line_list) != 2:
                    continue
                dnum = line_list[0].strip()
                temp_set_dnum.add(dnum)
                info_list = line_list[1].strip().split(",")

                if behav == "view" and len(info_list) != 102:
                    continue
                if behav != "view" and len(info_list) != 51:
                    continue
                if dnum not in dic_info:
                    dic_info[dnum] = [ int(info) for info in info_list]
            set_dnum = set_dnum | temp_set_dnum
            dic_behav_info[behav] = dic_info
            print("readDone " + channel + "-" + behav + " " + str(len(dic_info)))
            print("tempDnumSet:" + str(len(temp_set_dnum)))
            print("totalDnumSet:" + str(len(set_dnum)))
        dic_channel_behav_info[channel] = dic_behav_info


    # 生成矩阵，样本。
    print("-------------------")
    for channel,behav_info in dic_channel_behav_info.items():
        print(channel + " : " + str(len(behav_info)))
        for behav , info in behav_info.items():
            print(channel+"-"+behav+":"+str(len(info))  )
    print("-------------------")


    # 生成 infer 样本。

    writer = open("sample_infer", "w")
    for dnum in set_dnum:
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

        writer.write(",".join([ str(i) for i in dnum_total_info ]) + ":" + dnum + "\n")
        writer.flush()
    writer.flush()
    writer.close()
    return


gen_sample_infer()