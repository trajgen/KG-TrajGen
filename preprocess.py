import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from sklearn.model_selection import train_test_split
from helpers import encode_time
import csv
import re

'''
            将增强知识图谱的txt 文件转化为 csv文件   
'''

# 输入和输出文件名
input_filename = f'data/BJ_Taxi/UrbanKG_data/triplets_bj.txt'
output_filename = f'data/BJ_Taxi/UrbanKG_data/aug_knowledge_graph.csv'

# 打开文件
with open(input_filename, 'r', encoding='utf-8') as infile, open(output_filename, 'w', newline='',
                                                                 encoding='utf-8') as outfile:
    # 创建一个csv写入器
    writer = csv.writer(outfile)

    # 写入标题行（可选）
    writer.writerow(['head_entity', 'relation', 'tail_entity'])

    # 逐行读取txt文件
    for line in infile:
        # 去除行尾的换行符，并按空格分割成字段
        fields = line.strip().split()

        # 检查是否有三个字段
        if len(fields) == 3:
            # 写入csv文件
            writer.writerow(fields)
        else:
            # 如果不是三个字段，可以选择打印警告或跳过该行
            print(f"Warning: Skipping line with {len(fields)} fields: {line.strip()}")

print(f"Conversion complete. {output_filename} has been created.")


'''
    得到路段ID到实体ID的映射关系
'''
entity2id_file=f"./data/BJ_Taxi/UrbanKG_data/entity2id_bj.txt"
roadID2uID = np.full((37684, 1), 9999999)

with open(entity2id_file,"r") as lines:
    for line in lines:
        temp=line.split()
        parts=temp[0].split("/")
        name=parts[0]
        rID=parts[1]
        if name=="Road":
            roadID2uID[int(rID)]=int(temp[1])

#转化成csv文件
df = pd.DataFrame(roadID2uID)
df = df.rename(columns={0: 'KG_id'})
df = df.assign(road_ID=range(len(df)))
df.to_csv(f"./data/BJ_Taxi/UrbanKG_data/kgID2roadID.csv",index=False)






