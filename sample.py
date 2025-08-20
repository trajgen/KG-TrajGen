import argparse
import json
import os
import socket
import sys

import helpers
import numpy as np
import pandas as pd
import torch
import dgl
from model import MyTransformer, MyTransformerConfig
from torch.profiler import ProfilerActivity, profile, record_function
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=0, type=int, choices=[0, 1, 2, 3, 4]) ####0
parser.add_argument("--cuda", default="0", type=str)
parser.add_argument("--dtype", default="float32", type=str)
parser.add_argument("--method", default="STEGA", type=str)
parser.add_argument("--load_type", default="best", choices=["best", "last"])
parser.add_argument(
    "--data", type=str, default="BJ_Taxi", choices=["BJ_Taxi", "Porto_Taxi"]
)
parser.add_argument("--out_dir", default="out", type=str)
parser.add_argument("--num_samples", default=5000, type=int)
parser.add_argument("--top_k", default=5, type=int)
parser.add_argument("--init_from", default="resume", type=str)
parser.add_argument("--temperature", default="0.8", type=float)
parser.add_argument("--debug", default=False, type=helpers.str2bool)

# hyperparameter
parser.add_argument("--tf_head_num", default=2, type=int)
parser.add_argument("--tf_layer_num", default=2, type=int)
parser.add_argument("--gnn", default="r_gcn", type=str)
parser.add_argument("--gnn_layer_num", default=2, type=int)
parser.add_argument("--gnn_head_num", default=2, type=int)
args = parser.parse_args()

helpers.set_random_seed(args.seed)
args.hostname = socket.gethostname()
args.datapath = f"./data/{args.data}"

args.device = torch.device(
    f"cuda:{args.cuda}" if torch.cuda.is_available() else torch.device("cpu")
)
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[args.dtype]
args.ctx = torch.autocast(device_type="cuda", dtype=ptdtype)

args.out_dir = args.out_dir + "/main"

train_str = f"0-{args.gnn}-gly{args.gnn_layer_num}-ghd{args.gnn_head_num}-tly{args.tf_layer_num}-thd{args.tf_head_num}"

path_model_best = f"{args.out_dir}/{args.data}_{train_str}_ckpt_best.pth"
path_model_last = f"{args.out_dir}/{args.data}_{train_str}_ckpt_last.pth"
path_test_gene = f"{args.out_dir}/{args.data}_gen.csv"

log_dir = f"./logs"
log_prefix = (
    f"{args.method}-{args.data}-{train_str}-sample-{args.hostname}-gpu{args.cuda}"
)
logger = helpers.set_logger(log_dir=log_dir, log_prefix=log_prefix)
logger.info(args)

# load model
map_manager = helpers.MapManager(dataset_name=args.data)
if args.init_from == "resume":
    if args.load_type == "best":
        logger.info(f"load test model from {path_model_best}")
        ckpt = torch.load(path_model_best, map_location=args.device)
    else:
        logger.info(f"load test model from {path_model_last}")
        ckpt = torch.load(path_model_last, map_location=args.device)
    modelConfig = MyTransformerConfig(**ckpt["model_args"])
    modelConfig.device = args.device

    model = MyTransformer(modelConfig)

    state_dict = ckpt["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

else:
    logger.info("wrong assignment!")

model.eval()
model.to(args.device)

adj_mx = helpers.read_adjcent_file(args.data)
adj_dense = adj_mx.toarray()

'''
          use E-RKG      
'''
device="cuda:0"

augment_graph_filename = "./data/{}/UrbanKG_data/aug_knowledge_graph.csv".format(args.data)
augment_graph = pd.read_csv(augment_graph_filename)
augment_graph = augment_graph.values
g_urban = dgl.graph((augment_graph[:, 0], augment_graph[:, 2])).to(device)
etype_urban = torch.tensor(augment_graph[:, 1].reshape(-1)).to(device)


entity_embedding_filename = "./data/{}/UrbanKG_data/entity_new_kg_embedddings_gie_32.npy".format(args.data)
entity_embedding_mx = np.load(entity_embedding_filename)
entity_embedding_tensor = torch.tensor(entity_embedding_mx).to(torch.float32).to(device)


road = pd.read_csv(f"data/{args.data}/UrbanKG_data/kgID2roadID.csv")
road_KG_id = road.values
len_list=[]
road_embed=0



# generate trajectories
with torch.no_grad():
    with args.ctx:
        pred_data = model.generate(
            args,
            adj_dense,
            args.num_samples,g_urban,etype_urban,entity_embedding_tensor,road_KG_id,len_list,
            road_embed,
            temperature=args.temperature,
            top_k=args.top_k,
        )


f1 = open(path_test_gene, 'w')   #w
f1.write("rid_list,time_list,destination\n")

for i in range(len(pred_data)):
    f1.write('\"{}\",\"{}\",{}\n'.format(','.join([str(gps) for gps in pred_data[i][0]]),
                                     ','.join([str(gps) for gps in pred_data[i][1]]),
                                         int(pred_data[i][2])
                                     ))

