#!/usr/bin/python
# -*- coding: UTF-8 -*-
import happybase
import json
import pymysql
import sys
import time

# 连接 mysql
host = "10.81.244.140"
user = "middleoffice"
password = "1KXaVOYcDk933psB"
database = "middleoffice"
#host = "47.97.36.218"
#user = "datacenter"
#password = "Bds6NGda3lVaBNps"
#database = "datacenter"

thrift_host = "10.29.180.177"
hbase_table_name = "recommend_smallwindow"
row_key = "%s_tag_v1.0" % sys.argv[1]
key_rec_period1 = b"cf:recomInfoPeriod1"
key_rec_period2 = b"cf:recomInfoPeriod2"
key_create_time=b'cf:createTime'
connection = happybase.Connection(thrift_host)
table = connection.table(hbase_table_name)
row = table.row(row_key)
#print(sys.argv[1])
rec_period1 = row.get(key_rec_period1)
rec_period2 = row.get(key_rec_period2)
create_time = row.get(key_create_time).decode('utf-8')
create_date = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(int(create_time)/1000))
#rec_period1_dict=json.load(rec_period1.decode('utf-8'))
#rec_period2_dict=json.load(rec_period2.decode('utf-8'))
#print(rec_period1.decode('utf-8'))
#print(rec_period2.decode('utf-8'))
#print("fengjigang")

conn = pymysql.connect(host=host, user=user, password=password, database=database, charset="utf8")

# 得到一个可以执行SQL语句并且将结果作为字典返回的游标
cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)

new_dict = json.loads(rec_period1.decode('utf-8'))
print("==============================START====================================")
print('入库时间:%s' % create_date)
print('白天推荐:%s条' % len(new_dict["list"]))
for obj in new_dict["list"]:
    sql="SELECT cover_id,title,source_id,channel_name,operate_status FROM media_video_zt_detail  where source_id ='%s' and cover_id ='%s'" % (obj["source_id"],obj["album_id"])
    #print(obj["cid"],obj["source_id"])
    row_number = cursor.execute(sql)
    result = cursor.fetchall()

    for media in result:
       print("rec_period:%s,cover_id=%s,title=%s,source_id=%s,channel_name=%s,operate_status=%s" % ("day",media["cover_id"],media["title"],media["source_id"],media["channel_name"],media["operate_status"]))
print("==============================我是分割线====================================")
new_dict2 = json.loads(rec_period2.decode('utf-8'))
print('入库时间:%s' % create_date)
print('夜晚推荐:%s条' % len(new_dict2["list"]))
for obj in new_dict2["list"]:
    sql="SELECT cover_id,title,source_id,channel_name,operate_status FROM media_video_zt_detail  where source_id ='%s' and cover_id ='%s'" % (obj["source_id"],obj["album_id"])
    #print(obj["cid"],obj["source_id"])
    row_number = cursor.execute(sql)
    result = cursor.fetchall()

    for media in result:
       print("rec_period:%s,cover_id=%s,title=%s,source_id=%s,channel_name=%s,operate_status=%s" % ("night",media["cover_id"],media["title"],media["source_id"],media["channel_name"],media["operate_status"]))



# 关闭光标对象
cursor.close()
# 关闭数据库连接
conn.close()

