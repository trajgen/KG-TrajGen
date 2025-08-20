"""

The processed files are stored in ./processed_data
File format after alignment, filtering and preprocessing:
    borough.csv: borough_id, borough_name, borough_polygon
    area.csv: area_id, area_name, area_polygon
    poi.csv: poi_id, poi_name, poi_category, lat, lng, borough_id, area_id
    road.csv: road_id, road_name, road_category, from_junction, to_junction, road_polygon, lat, lng, borough_id, area_id
    junction.csv: junction_id, junction_catogory, lat, lng, borough_id, area_id

"""
from shapely import wkt
from tqdm import tqdm
import geopandas as gpd
import pandas as pd
import folium
import re
from shapely.geometry import MultiPolygon
import numpy as np
from shapely.geometry import Point
from shapely.coords import CoordinateSequence
from shapely.geometry import LineString

"""

    borough, area's multi-polygon latitude and longitude reading

"""

# bj_area 和 bj_borough
######################################
dataframe1 = gpd.read_file('./Meta_data/bj/Administrative_data/Borough/bj_borough.shp')
# 转换为经纬度坐标系
dataframe1 = dataframe1.to_crs('EPSG:4326')
seleceted_colums1 = ['borough_ID', 'name', 'geometry']
borough_dataframe = dataframe1[seleceted_colums1]
print(borough_dataframe.shape)
# borough_dataframe.to_csv('./Processed_data/bj/bj_borough.csv',index=False)
#####################################
dataframe2 = gpd.read_file('./Meta_data/bj/Administrative_data/Area/bj_function.shp')
# 转换为经纬度坐标系
dataframe2 = dataframe2.to_crs('EPSG:4326')  # to_crs（）：更改坐标参考系。
seleceted_colums2 = ['area_ID', 'name', 'function', 'geometry']
area_dataframe = dataframe2[seleceted_colums2]
## 过滤掉 1, 103, 104 区域因为他们在孤岛，或者特别小
# area_dataframe.to_csv('./Processed_data/bj/bj_area.csv',index=False)
######################################
#
"""

    Align POI to borough, area, judge whether POI point is inside polygon or multipolygon
    #poi与区域对齐，判断是否在这个多边形里面
    
    poi  后面加上borough_ID area_ID 
"""

# poi_dataframe = pd.read_csv('./Meta_data/bj/POI/bj_poi.csv')
# point = wkt.loads(poi_dataframe['geometry'])
# poi_datanumpy = []
# for data in point:
#     poi_datanumpy.append([data.x, data.y])
#
# poi_datanumpy = np.array(poi_datanumpy)
# # 获取POI的经纬度信息
# # borough, area 矩阵
# poi_borough_area_id = np.full((poi_datanumpy.shape[0], 2), -1)
# # 用999填充      形状为（poi个数，2）
# # 这里有两个循环，第一个循环是遍历poi点，获取到poi点之后，
# for i in tqdm(range(poi_datanumpy.shape[0])):  # tqdm是进度条   遍历每个poi
#     poi_point = Point(poi_datanumpy[i][0], poi_datanumpy[i][1])  # 获取点信息
#     ## 遍历 borough
#     for j in range(borough_dataframe.shape[0]):  #
#         borough_polygon = borough_dataframe.iloc[j].geometry  # 得到每个区域的经纬度信息
#         if borough_polygon.contains(poi_point):
#             poi_borough_area_id[i][0] = borough_dataframe.iloc[j].borough_ID
#             break
#     ## 遍历 area
#     for k in range(area_dataframe.shape[0]):
#         area_polygon = area_dataframe.iloc[k].geometry
#         if area_polygon.contains(poi_point):
#             poi_borough_area_id[i][1] = area_dataframe.iloc[k].area_ID
#             break
#
# # Add to the dataframe as a new column and filter
# poi_dataframe[['borough_id', 'area_id']] = poi_borough_area_id
# # dsf=poi_dataframe['region_id'] != 999
# poi_dataframe = poi_dataframe[(poi_dataframe['borough_id'] != -1)]
# poi_dataframe = poi_dataframe[(poi_dataframe['area_id'] != -1)]
# poi_dataframe.to_csv('./Processed_data/bj/bj_poi.csv',index=False)
#
# ######################################

"""

    The road is aligned to the borough, area, to determine whether the road is inside the Polygon, or a small part of the road is inside the Polygon
        路段与区域对齐，去决定这个路段是否在这个多边形里面
    
    路段    后面加上borough_ID area_ID 
    
"""
#
road_dataframe = pd.read_csv('./Meta_data/bj/RoadNetwork/road_segment/roadmap.csv')
# road_datanumpy = road_dataframe[['coordinates']].values  #,values是转化成numnpy数组
road_datanumpy = wkt.loads(road_dataframe['geometry']).values.reshape(road_dataframe.shape[0], 1)
# borough, area
road_borough_area_id = np.full((road_datanumpy.shape[0], 2), -1)
for i in tqdm(range(road_datanumpy.shape[0])):
    # sad=road_datanumpy[i][0]
    # coordseq = CoordinateSequence(road_datanumpy[i][0])
    road_linestring = road_datanumpy[i][0]
    center = road_linestring.centroid  # 这里将线转点 确保每个路段都会有borough和areaID
    ##判断路段是否在borough内
    for j in range(borough_dataframe.shape[0]):
        borough_polygon = borough_dataframe.iloc[j].geometry
        if borough_polygon.contains(center) or borough_polygon.boundary.intersects(center):
            road_borough_area_id[i][0] = borough_dataframe.iloc[j].borough_ID
            break
    ##判断路段是否在area内
    for j in range(area_dataframe.shape[0]):
        area_polygon = area_dataframe.iloc[j].geometry
        if area_polygon.contains(center) or area_polygon.boundary.intersects(center):
            road_borough_area_id[i][1] = area_dataframe.iloc[j].area_ID
            break


# road_dataframe = road_dataframe[(road_dataframe['borough_id'] != 9999)]
# road_dataframe = road_dataframe[(road_dataframe['area_id'] != 9999)]

for i in tqdm(range(road_datanumpy.shape[0])):
    if road_borough_area_id[i][1]==-1:
        road_linestring = road_datanumpy[i][0]
        center = road_linestring.centroid
        min_distance = float('inf')  # 初始化最小距离为无穷大
        nearest_area_id = None  # 初始化最近区域的 ID 为 None

        # 遍历每个区域多边形
        for j in range(area_dataframe.shape[0]):
            area_polygon = area_dataframe.iloc[j].geometry  # 获取当前区域的多边形对象
            distance = center.distance(area_polygon)  # 计算质心到多边形边界的最小距离

            # 如果当前距离比之前的最小距离小，则更新最小距离和最近区域的 ID
            if distance < min_distance:
                min_distance = distance
                nearest_area_id = area_dataframe.iloc[j].area_ID

                # 将最近区域的 ID 分配给当前道路
        road_borough_area_id[i][1] = nearest_area_id


road_dataframe[['borough_id', 'area_id']] = road_borough_area_id
road_dataframe.to_csv('./Processed_data/bj/roadmap_2.csv', index=False)

# ######################################
"""

    The junction is aligned to the borough, area, and it is judged whether the junction is inside the Polygon
     路口对齐  
     判断路口再在哪个区域内   
     路口  后面加上borough_ID area_ID 
"""
# junction_dataframe = pd.read_csv('./Meta_data/bj/RoadNetwork/road_junction/bj_junction.csv')
# junction = wkt.loads(junction_dataframe['geometry'])
#
# junction_datanumpy = []
#
# for data in junction:
#     junction_datanumpy.append([data.x, data.y])
#
# junction_datanumpy = np.array(junction_datanumpy)
#
# # borough, area 矩阵
# junction_borough_area_id = np.full((junction_datanumpy.shape[0], 2), -1)
#
# for i in tqdm(range(junction_datanumpy.shape[0])):
#
#     junction_point = Point(junction_datanumpy[i][0], junction_datanumpy[i][1])
#     ##
#     for j in range(borough_dataframe.shape[0]):
#         borough_polygon = borough_dataframe.iloc[j].geometry
#         if borough_polygon.contains(junction_point):
#             junction_borough_area_id[i][0] = borough_dataframe.iloc[j].borough_ID
#             break
#     ##
#     for k in range(area_dataframe.shape[0]):
#         area_polygon = area_dataframe.iloc[k].geometry
#         if area_polygon.contains(junction_point):
#             junction_borough_area_id[i][1] = area_dataframe.iloc[k].area_ID
#             break
#
# # Add to the dataframe as a new column and filter
# junction_dataframe[['borough_id', 'area_id']] = junction_borough_area_id
# junction_dataframe = junction_dataframe[(junction_dataframe['borough_id'] != -1)]
# junction_dataframe = junction_dataframe[(junction_dataframe['area_id'] != -1)]
#
# junction_dataframe.to_csv('./Processed_data/bj/bj_junction.csv',index=False)