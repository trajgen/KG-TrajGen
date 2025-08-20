
from shapely import wkt
from tqdm import tqdm
import geopandas as gpd
import pandas as pd
import folium
import re
from shapely.geometry import MultiPolygon
import numpy as np
from shapely.geometry import Point
#####################################
# load data
dataframe1 = gpd.read_file('./Meta_data/bj/Administrative_data/Borough/bj_borough.shp')
# 转换为经纬度坐标系
dataframe1 = dataframe1.to_crs('EPSG:4326')  #wgs84
seleceted_colums1 = ['borough_ID', 'name', 'geometry']
borough_dataframe = dataframe1[seleceted_colums1]
######################################
dataframe2 = gpd.read_file('./Meta_data/bj/Administrative_data/Area/bj_function.shp')
# Convert to latitude-longitude coordinate system
dataframe2 = dataframe2.to_crs('EPSG:4326')   #to_crs（）：更改坐标参考系。
seleceted_colums2 = ['area_ID', 'name','function', 'geometry']
area_dataframe = dataframe2[seleceted_colums2]
#####################################
"""

Relation 1: Borough Nearby Borough BNB

# """
###################################

BNB = []
for i in tqdm(range(borough_dataframe.shape[0])):
    head_borough = borough_dataframe.iloc[i].geometry
    for j in range(borough_dataframe.shape[0]):
        tail_borough = borough_dataframe.iloc[j].geometry
        if head_borough.touches(tail_borough):
            BNB.append('Borough/' + str(borough_dataframe.iloc[i].borough_ID) + ' BNB ' + 'Borough/' + str(borough_dataframe.iloc[j].borough_ID))

#####################################

"""

Relation 2: Area Nearby Area ANA

"""

#####################################

ANA = []
for i in tqdm(range(area_dataframe.shape[0])):
    head_area = area_dataframe.iloc[i].geometry
    for j in range(area_dataframe.shape[0]):
        tail_area = area_dataframe.iloc[j].geometry
        if head_area.touches(tail_area):
            ANA.append('Area/' + str(area_dataframe.iloc[i].area_ID) + ' ANA ' + 'Area/' + str(area_dataframe.iloc[j].area_ID))

#####################################

"""

Relation 3: POI Locates at Area PLA
Relation 4: POI Belongs to Borough PBB
Relation 5: POI Has POI Category PHPC

"""

#####################################

PLA = []
PBB = []
PHPC = []

poi_dataframe = pd.read_csv('./Processed_data/bj/modified_bj_poi.csv')
poi_datanumpy = np.array(poi_dataframe[[ "poi_ID", "borough_id", "area_id", "big_type"]])

for i in tqdm(range(poi_datanumpy.shape[0])):
    PBB.append('POI/' + str(poi_datanumpy[i][0]) + ' PBB '
               + 'Borough/' + str(poi_datanumpy[i][1]))

for i in tqdm(range(poi_datanumpy.shape[0])):
    PLA.append('POI/' + str(poi_datanumpy[i][0]) + ' PLA '
               + 'Area/' + str(poi_datanumpy[i][2]))

for i in tqdm(range(poi_datanumpy.shape[0])):
    PHPC.append('POI/' + str(poi_datanumpy[i][0]) + ' PHPC '
               + 'PC/' +str(poi_datanumpy[i][3]))

####################################

"""

Relation 6: Road Locates at Area RLA
Relation 7: Road Belongs to Borough RBB
Relation 8: Road Has Road Category RHRC

"""

#####################################

RLA = []
RBB = []
RHRC = []

road_dataframe = pd.read_csv('./Processed_data/bj/roadmap_2.csv')
road_datanumpy = np.array(road_dataframe[[ "geo_id", "borough_id", "area_id","highway"]])


for i in tqdm(range(road_datanumpy.shape[0])):
    RBB.append('Road/' + str(road_datanumpy[i][0]) + ' RBB '
               + 'Borough/' + str(road_datanumpy[i][1]))

for i in tqdm(range(road_datanumpy.shape[0])):
    RLA.append('Road/' + str(road_datanumpy[i][0]) + ' RLA '
               + 'Area/' + str(road_datanumpy[i][2]))

for i in tqdm(range(road_datanumpy.shape[0])):
    RHRC.append('Road/' + str(road_datanumpy[i][0]) + ' RHRC '
               + 'RC/'+ 'highway' + str(road_datanumpy[i][3]))

###############################################

"""

Relation 9: Junction Locates at Area JLA
Relation 10: Junction Belongs to Borough JBB
Relation 11: Junction Has Junction Category JHJC

"""

#####################################

JLA = []
JBB = []
JHJC = []

junction_dataframe = pd.read_csv('./Processed_data/bj/bj_junction.csv')
junction_datanumpy = np.array(junction_dataframe[[ "jun_ID", "borough_id", "area_id", "fclass"]])

for i in tqdm(range(junction_datanumpy.shape[0])):
    JBB.append('Junction/' + str(junction_datanumpy[i][0]) + ' JBB '
               + 'Borough/' + str(junction_datanumpy[i][1]))

for i in tqdm(range(junction_datanumpy.shape[0])):
    JLA.append('Junction/' + str(junction_datanumpy[i][0]) + ' JLA '
               + 'Area/' + str(junction_datanumpy[i][2]))

for i in tqdm(range(junction_datanumpy.shape[0])):
    JHJC.append('Junction/' + str(junction_datanumpy[i][0]) + ' JHJC '
               + 'JC/' + str(junction_datanumpy[i][3]))

####################################

"""

Relation 12：Junction Belongs to Road JBR

"""



# #####################################
JBR = []
distance_set=[]

road_dataframe = pd.read_csv('./Processed_data/bj/roadmap.csv')
junction_dataframe2 = pd.read_csv('./Processed_data/bj/bj_junction_raw.csv')
road_datanumpy = road_dataframe[['geo_id']].values
junction_datanumpy2 = junction_dataframe2[['jun_ID']].values

road_cood=wkt.loads(road_dataframe['geometry']).values.reshape(road_dataframe.shape[0],1)
junctiopn_cood=wkt.loads(junction_dataframe2['geometry']).values.reshape(junction_dataframe2.shape[0],1)




for i in tqdm(range(junction_datanumpy2.shape[0])):
    junction_point=junctiopn_cood[i][0]
    for j in (range(road_datanumpy.shape[0])):
        road_linestring=road_cood[j][0]
        distance=road_linestring.distance(junction_point)
        distance_set.append(distance)
        if distance<1e-8:
            JBR.append('Junction/' + str(junction_datanumpy2[i][0]) + ' JBR '
                       + 'Road/' + str(road_datanumpy[j][0]))
            break


# print(len(JBR))
# #####################################

"""

Relation 13：Area Locates at Borough ALB

"""

#####################################
ALB = []

for i in tqdm(range(area_dataframe.shape[0])):
    area = area_dataframe.iloc[i].geometry
    for j in range(borough_dataframe.shape[0]):
        borough = borough_dataframe.iloc[j].geometry
        if area.within(borough) or area.intersects(borough):
            ALB.append('Area/' + str(area_dataframe.iloc[i].area_ID) + ' ALB ' + 'Borough/' + str(borough_dataframe.iloc[j].borough_ID))
            break

####################################



base_KG=[]
with open(r'./UrbanKG/bj/UrbanKG_base_bj.txt','r') as f1:
    for line in f1:
        base_KG.append(line.strip())

f1.close()
base_KG.extend(PLA)
base_KG.extend(RLA)
base_KG.extend(JLA)
base_KG.extend(PBB)
base_KG.extend(RBB)
base_KG.extend(JBB)
base_KG.extend(ALB)
base_KG.extend(JBR)
base_KG.extend(BNB)
base_KG.extend(ANA)
base_KG.extend(PHPC)
base_KG.extend(RHRC)
base_KG.extend(JHJC)

with open(r'./UrbanKG/bj_2/UrbanKG_augmented_bj.txt','w') as f2:
    for i in range(len(base_KG)):
        f2.write(base_KG[i])
        f2.write('\n')

f2.close()