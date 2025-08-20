from shapely import wkt
from tqdm import tqdm
import geopandas as gpd
import pandas as pd
import folium
import re
from shapely.geometry import MultiPolygon
import numpy as np
from shapely.geometry import Point
from tqdm import tqdm
from osm_poi_category_porto import cates,cates_labels
"""

    borough, area's multi-polygon latitude and longitude reading

"""

# # 处理road_junction类别
# ######################################
dataframe1 = gpd.read_file('./Meta_data/bj/RoadNetwork/road_junction/bj_junction.shp')
# 转换为经纬度坐标系
dataframe1 = dataframe1.to_crs('EPSG:4326')
seleceted_colums1 = ['jun_ID','osm_id', 'code','fclass','geometry']
poi_dataframe1 = dataframe1[seleceted_colums1]
#
#
#
#
poi_dataframe1.to_csv("./Meta_data/bj/RoadNetwork/road_junction/bj_junction.csv",index=False)

# ##添加road信息
# from pyproj import CRS
#
# # 读取包含几何对象的文件
# data = gpd.read_file("./Meta_data/porto/RoadNetwork/porto_road_2022_3.shp")
#
# # 定义目标投影坐标系
# target_crs = CRS.from_epsg(3857)  # 例如，使用WGS 84 Web Mercator投影坐标系
#
# # 将几何对象转换为目标投影坐标系
# data = data.to_crs(target_crs)

# # 进行长度计算
# lengths = data.geometry.length.tolist()
#
# lengths=np.array(lengths).reshape(len(lengths),1)
# road_data=pd.read_csv("./Meta_data/porto/RoadNetwork/porto_road2.csv")
# road_data[['length']]=lengths


# # # 处理road类别
# # ######################################
# dataframe1 = gpd.read_file('./UrbanKG_data/porto_taxi_data_11095/shp/porto_geo.shp')
# # # 转换为经纬度坐标系
# dataframe1 = dataframe1.to_crs('EPSG:4326')
# seleceted_colums1 = ['name','ID','osm_id', 'code','fclass','geometry','maxspeed','oneway','tunnel','bridge']
# poi_dataframe1 = dataframe1[seleceted_colums1]
# # # print(type(lengths))
# # # print(len(lengths))
# # # print(len(poi_dataframe1))
# # # poi_dataframe1[['length']]=lengths
# # #
# # # #
# # # #
# # # #
# # # #
# road_data.to_csv("./Meta_data/porto/RoadNetwork/porto_road2.csv",index=False)


'''
       处理geo路段

'''

dataframe1 = gpd.read_file('./Meta_data/bj/RoadNetwork/road_segment/roadmap.shp')
# 转换为经纬度坐标系
dataframe1 = dataframe1.to_crs('EPSG:4326')
seleceted_colums1 = ['geo_id','highway','oneway','name','length','geometry']
geo_dataframe1 = dataframe1[seleceted_colums1]




geo_dataframe1.to_csv("./Meta_data/bj/RoadNetwork/road_segment/roadmap.csv",index=False)










