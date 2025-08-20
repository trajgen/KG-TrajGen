from shapely import wkt
from tqdm import tqdm
import geopandas as gpd
import pandas as pd
import folium
import re
from shapely.geometry import MultiPolygon
import numpy as np
from shapely.geometry import Point
from shapely.wkt import loads
from geopy import distance
import os
'''
                       1、Road connectby Road
                        2、Road FlowTransition Road
                        3、Road SpatialSimilarity Road 
                        4、Road TimeSimilarity Road 
'''



#load data

road_num=10904
dataset="Porto"
road_rel=pd.read_csv("./Processed_data/{}/porto.rel".format(dataset))
poi_dataframe = pd.read_csv('./Processed_data/{}/porto_poi.csv'.format(dataset))
road_dataframe = pd.read_csv('./Processed_data/{}/porto_road_geo.csv'.format(dataset))
traj_dataframe = pd.read_csv('./Processed_data/{}/traj_tra.csv'.format(dataset))       #读取轨迹数据


'''
                relation 1: Road connectby Road     10904
'''

RCR=[]

rel_datanumpy=np.array(road_rel[["origin_id","destination_id"]])
for i in tqdm(range(rel_datanumpy.shape[0]),desc="cal road rel"):
    RCR.append('Road/' + str(rel_datanumpy[i][0]) + ' RCR '
               + 'Road/' + str(rel_datanumpy[i][1]))



'''
                relation 2：Road SpatialSimilaty Road
'''


RSR=[]
top_n=3
distance_threshold=100
poi_datanumpy = np.array(poi_dataframe[[ "poi_ID", "geometry","cate"]])
road_datanumpy = np.array(road_dataframe[[ "geo_id", "geometry"]])
poi_categories = poi_dataframe['cate'].unique()

def cosine_similarity(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 and norm2 == 0:
        return 1.
    elif norm1==0 or norm2==0:
        return -2.
    dot_product = np.dot(vec1, vec2)
    similarity = dot_product / (norm1 * norm2)
    return similarity
#

def calculate_poi_distribution(road_id):

    road_line=loads(road_datanumpy[road_id][1])
    road_center = road_line.centroid

    poi_gdf = gpd.GeoDataFrame(poi_dataframe)



    poi_distribution = pd.Series(0,index=poi_categories, dtype=int)


    for index, row in poi_gdf.iterrows():
        poi_geometry = loads(row.geometry)
        d = distance.distance((poi_geometry.y, poi_geometry.x),
                              (road_center.y, road_center.x)).meters
        if d <= distance_threshold:
            category = row['cate']
            poi_distribution[category] += 1

    poi_distribution_array = poi_distribution.values

    return poi_distribution_array

road_poi_vector_file="./Processed_data/{}/road_poi_vector_mx.npy".format(dataset)
if os.path.exists(road_poi_vector_file):
    road_poi_distribution = np.load(road_poi_vector_file)
else:
    road_poi_distribution = np.zeros((road_num, poi_categories.shape[0]))
    for i in tqdm(range(road_datanumpy.shape[0]), desc="cal POI vector"):
        road_i_poi_vector = calculate_poi_distribution(i)
        road_poi_distribution[i] = road_i_poi_vector

    np.save(road_poi_vector_file, road_poi_distribution)



sim_numpy=np.full((road_num,road_num),-2,dtype=np.float32)

for i in tqdm(range(road_datanumpy.shape[0]),desc="cal similaty mx"):
    road_i_poi_vector=road_poi_distribution[i]
    road_poi_num=sum(road_i_poi_vector)
    if road_poi_num==0:
        continue
    for j in range(road_datanumpy.shape[0]):
        if i!=j:
            road_j_poi_vector=road_poi_distribution[j]
            road_cosine=cosine_similarity(road_i_poi_vector,road_j_poi_vector)
            sim_numpy[i][j]=road_cosine



for index,row in enumerate(sim_numpy):

    road_i_poi_vector=road_poi_distribution[index]
    road_poi_num=sum(road_i_poi_vector)
    if road_poi_num==0:
        continue

    sorted_indices = np.argsort(row)[::-1]
    top_n_indices = sorted_indices[:top_n]
    for k in top_n_indices:
        RSR.append('Road/' + str(index) + ' RSR '
                   + 'Road/' + str(k))



'''
                relation 3：Road TemporalSimilaty Road                   
'''

RTR=[]
top=3
tem_sim_numpy=np.full((road_num,road_num),-2,dtype=np.float32)

road_time_distribution=np.zeros((road_num,2880))
for index,row in tqdm(traj_dataframe.iterrows(),total=traj_dataframe.shape[0],desc="cal temporal similaty"):
    rid_list = [int(i) for i in row["rid_list"].split(',')]
    time_list = [int(i) for i in row["time_encode"].split(',')]
    for index in range(len(rid_list)):
        road_time_distribution[rid_list[index]][time_list[index]]+=1


#计算时间相似度矩阵
for i in tqdm(range(road_num),desc="cal time similaty mx"):
    road_i_time_vector=road_time_distribution[i]
    road_poi_num=sum(road_i_time_vector)
    if road_poi_num==0:
        print(i)
        continue
    for j in range(road_num):
        if i!=j:
            road_j_time_vector=road_time_distribution[j]
            road_cosine=cosine_similarity(road_i_time_vector,road_j_time_vector)
            tem_sim_numpy[i][j]=road_cosine



for index,row in enumerate(tem_sim_numpy):

    road_i_poi_vector=road_time_distribution[index]
    road_poi_num=sum(road_i_poi_vector)
    if road_poi_num==0:
        continue

    sorted_indices = np.argsort(row)[::-1]
    top_n_indices = sorted_indices[:top]
    for k in top_n_indices:
        RTR.append('Road/' + str(index) + ' RTR '
                   + 'Road/' + str(k))




'''
                relation 4：Road FlowTransition Road
'''

RFR=[]
history_transfer_mx=np.zeros([road_num,road_num])
for index,row in tqdm(traj_dataframe.iterrows(),total=traj_dataframe.shape[0],desc="cal transfer mx"):
    rid_list=[int(i) for i in row["rid_list"].split(',')]
    for i in range(len(rid_list)-1):
        history_transfer_mx[rid_list[i]][rid_list[i+1]]+=1

max_indices=np.argmax(history_transfer_mx,axis=1)
for i in range(road_num):
    s=sum(history_transfer_mx[i][:])
    if s==0:
        continue
    RFR.append('Road/' + str(i) + ' RFR '
               + 'Road/' + str(max_indices[i]))



'''
                relation 5：Road ODflow Road                 
'''

# ROR=[]
# top_od=3
# od_transfer_mx=np.zeros([road_num,road_num])
#
# for index,row in tqdm(traj_dataframe.iterrows(),total=traj_dataframe.shape[0],desc="cal od transfer mx"):
#     rid_list=[int(i) for i in row["rid_list"].split(',')]
#     orign=rid_list[0]
#     des=rid_list[-1]
#     od_transfer_mx[orign][des]+=1
#
# # column_sums = np.sum(od_transfer_mx, axis=0)
# # hang_sums = np.sum(od_transfer_mx, axis=1)
# # maxsa=np.argmax(column_sums)
# # column_sums_2 = np.sum(column_sums, axis=0)
# top_od_indices = np.array([np.argsort(row)[::-1][:top_od] for row in od_transfer_mx])  #取出前n个索引
#
#
# for i in range(road_num):
#     for j in range(top_od):
#         if od_transfer_mx[i][top_od_indices[i][j]] ==0:
#             continue
#         ROR.append('Road/' + str(i) + ' ROR '
#                    + 'Road/' + str(top_od_indices[i][j]))



total_num=len(RCR)+len(RSR)+len(RFR)+len(RTR)
print("RCR num：,RSR num：,RFR num：,RTR num：,total num：",len(RCR),len(RSR),len(RFR),len(RTR),total_num)



RCR.extend(RSR)
RCR.extend(RTR)
RCR.extend(RFR)


with open(r'UrbanKG/porto_10904/new_KG/UrbanKG_base_porto_2.txt', 'w') as f2:
    for i in range(len(RCR)):
        f2.write(RCR[i])
        f2.write('\n')

f2.close()




































