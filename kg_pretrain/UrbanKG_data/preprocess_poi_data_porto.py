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

# # 处理POI类别
# ######################################
dataframe1 = gpd.read_file('./Meta_data/bj/POI/bj_end_poi.shp')
# 转换为经纬度坐标系
dataframe1 = dataframe1.to_crs('EPSG:4326')
seleceted_colums1 = ['name','poi_ID','adname', 'type','type_lst_l','FID_bj_fun','geometry']
poi_dataframe1 = dataframe1[seleceted_colums1]




poi_dataframe1.to_csv("./Meta_data/bj/POI/bj_poi.csv",index=False)
#
#
#
# porto_poi_csv=pd.read_csv("./Meta_data/porto/POI/porto_poi.csv")
# poi_category=set()
#
# for index,row in tqdm(porto_poi_csv.iterrows(),total=len(porto_poi_csv),desc='count poi_cate'):
#     poi_category.add(row["fclass"])
#
# with open("output.txt", "w") as file:
#     for item in poi_category:
#         file.write(item + "\n")
#
# s=0
# for i in cates:
#     s+=len(cates_labels[i])   #计算类别的总长度
#     for value in cates_labels[i]:
#         if value.lower() not in poi_category:  #osm_poi_category_porto.py是否有多的类别
#             print(value+' ')
# print(s)
#
# poi_category=list(poi_category)
# for i in poi_category:
#     k=1
#     for j in range(12):  #11是合并类别总数
#          for values in cates_labels[cates[j]]:
#              if i==values.lower():
#                 k=0
#                 break
#     if k==1:
#         print(i)
#
#
#
#
#
# porto_poi_csv.to_csv("./Meta_data/porto/POI/porto_poi.csv")









