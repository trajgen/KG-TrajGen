import argparse
import json
import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.optim
import pandas as pd
import numpy as np
import models
import optimizers.regularizers as regularizers
from datasets.kg_dataset import KGDataset
from models import all_models
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

DATA_PATH = './data'

#命令行参数
parser = argparse.ArgumentParser(
    description="Urban Knowledge Graph Embedding"
)
parser.add_argument(
    "--dataset", default="bj_xiaorong_wo_area", choices=["NYC", "CHI","porto",'bj','bj_2'],
    help="Urban Knowledge Graph dataset"
)
parser.add_argument(
    "--model", default="GIE", choices=all_models, help='"TransE", "CP", "MurE", "RotE", "RefE", "AttE",'
                                                       '"ComplEx", "RotatE",'
                                                       '"RotH", "RefH", "AttH"'
                                                       '"GIE'
)
parser.add_argument(
    "--optimizer", choices=["Adagrad", "Adam", "SparseAdam"], default="Adam",
    help="Optimizer"
)
parser.add_argument(
    "--max_epochs", default=300, type=int, help="Maximum number of epochs to train for"
)
parser.add_argument(
    "--patience", default=30, type=int, help="Number of epochs before early stopping"
)
parser.add_argument(
    "--valid", default=3, type=float, help="Number of epochs before validation"
)
parser.add_argument(
    "--rank", default=32, type=int, help="Embedding dimension"    #嵌入维度32
)
parser.add_argument(
    "--batch_size", default=4096, type=int, help="Batch size"     #批量大小为4120
)
parser.add_argument(
    "--learning_rate", default=1e-3, type=float, help="Learning rate"
)
parser.add_argument(
    "--neg_sample_size", default=50, type=int, help="Negative sample size, -1 to not use negative sampling"
)
parser.add_argument(
    "--init_size", default=1e-3, type=float, help="Initial embeddings' scale"
)
parser.add_argument(
    "--multi_c", action="store_false", help="Multiple curvatures per relation"
)
parser.add_argument(
    "--regularizer", choices=["N3", "F2"], default="N3", help="Regularizer"
)
parser.add_argument(
    "--reg", default=0, type=float, help="Regularization weight"
)
parser.add_argument(
    "--dropout", default=0, type=float, help="Dropout rate"   #暂退法
)
parser.add_argument(
    "--gamma", default=0, type=float, help="Margin for distance-based losses"
)
parser.add_argument(
    "--bias", default="constant", type=str, choices=["constant", "learn", "none"],
    help="Bias type (none for no bias)"
)
parser.add_argument(
    "--dtype", default="double", type=str, choices=["single", "double"], help="Machine precision"
)
parser.add_argument(
    "--double_neg", action="store_true",
    help="Whether to negative sample both head and tail entities"
)
parser.add_argument(
    "--debug", action="store_true",
    help="Only use 1000 examples for debugging"
)


def get_embeddings(args):
    # create model
    dataset_path = os.path.join(DATA_PATH, args.dataset) #porto
    dataset = KGDataset(dataset_path, args.debug)
    args.sizes = dataset.get_shape()

    model = getattr(models, args.model)(args) #加载模型
    model.load_state_dict(torch.load(
        os.path.join("./logs/08_04/bj_xiaorong_wo_area/GIE_16_26_55/model.pt")))
    entity_embeddings = model.entity.weight.detach().numpy()     #这里是训练出来的实体嵌入
    #entity_idx_embedding 在实体和学习到的嵌入之间构建映射关系  ，不是什么实体的初始向量
    idx = pd.read_csv(DATA_PATH + '/' + args.dataset + "/entity_idx_embedding.csv", header=None)
    entity_idx = np.array(idx)

    entity_final_embedddings = np.zeros([entity_embeddings.shape[0], entity_embeddings.shape[1]])
    for i in tqdm(range(entity_embeddings.shape[0])):

        entity_final_embedddings[int(entity_idx[i])] = entity_embeddings[i]


    return entity_final_embedddings      #总之，获得所有实体的嵌入


def get_region_embeddings(grid_KG_id_path, entity_final_embedddings, save_path):

    grid = pd.read_csv(grid_KG_id_path)
    grid_KG_id = grid[["region_id", "KG_id"]].values
    grid_embeddings = np.zeros([260, 32])

    for i in range(grid_embeddings.shape[0]):
        grid_embeddings[i] = entity_final_embedddings[int(grid_KG_id[i][1])]

    print(grid_embeddings)
    np.save(save_path, grid_embeddings)



def get_POI_embedding(grid_KG_id_path, entity_final_embedddings, save_path):
    poi = pd.read_csv(grid_KG_id_path)
    poi_KG_id = poi[["poi_id", "KG_id", "Region_id"]].values
    poi_embeddings = np.zeros([1600, 33])
    for i in range(poi_embeddings.shape[0]):
        poi_embeddings[i] [0:32] = entity_final_embedddings[int(poi_KG_id[i][1])]
        poi_embeddings[i] [32] =  int(poi_KG_id[i][2])

    print(poi_embeddings)
    np.save(save_path, poi_embeddings)

'''
          得到所有实体嵌入中路段的嵌入
'''
def get_Road_embedding(grid_KG_id_path, entity_final_embedddings, save_path):
    road_num=10904
    road = pd.read_csv(grid_KG_id_path)
    road_KG_id = road[["road_ID", "KG_id"]].values
    road_embeddings = np.zeros([road_num, args.rank])
    for i in range(road_embeddings.shape[0]):
        road_embeddings[i] [0:32] = entity_final_embedddings[int(road_KG_id[i][1])]

    np.save(save_path, road_embeddings)

save_path="./result/entity_new_kg_embedddings_bj_GIE_wo_area.npy"                                                                    #解析参数
entity_final_embedddings = get_embeddings(parser.parse_args())
np.save(save_path,entity_final_embedddings)

# get_region_embeddings("xxx.csv", entity_final_embedddings,
#                       'xxx.npy')
#
# get_POI_embedding("xxx.csv", entity_final_embedddings,
#                       'xxx.npy')

# get_Road_embedding("./result/road_embedding.csv", entity_final_embedddings,
#                       './result/road_embedding.npy')















