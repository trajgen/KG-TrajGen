import argparse
import os
import re
from math import ceil
from pathlib import Path
import socket
import numpy as np
import pandas as pd
import dgl
import torch
from scipy.sparse import coo_matrix
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import helpers
from tqdm import tqdm
from model import MyTransformerConfig, MyTransformer
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("./tensorboard/porto_3/train")
writer2 = SummaryWriter("./tensorboard/porto_3/valid")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--cuda", default="0", type=str)
    parser.add_argument("--dtype", default="float32", type=str)
    parser.add_argument("--method", default="STEGA", type=str)
    parser.add_argument(
        "--data",
        type=str,
        default="Porto_Taxi",
        choices=["BJ_Taxi", "Porto_Taxi"],
    )
    parser.add_argument("--datapath", default="", type=str)
    parser.add_argument("--out_dir", default="out", type=str)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--vocab_size", default=0, type=int)

    # gnn settings
    parser.add_argument("--embed_dim", default=256, type=int)
    parser.add_argument("--gps_emb_dim", default=10, type=int)
    parser.add_argument("--gnn_layer_num", default=2, type=int)
    parser.add_argument("--gnn_head_num", default=2, type=int)
    parser.add_argument("--gnn", default="r_gcn", type=str)

    parser.add_argument("--lr", type=float, default=5e-4)  #5e-4
    parser.add_argument("--weight_decay", type=float, default=0) #1e-4
    parser.add_argument("--lr_patience", type=float, default=2)
    parser.add_argument("--lr_decay_ratio", type=float, default=1e-1)
    parser.add_argument("--early_stop_lr", type=float, default=1e-6)
    parser.add_argument("--batch_size", default=32, type=int)

    # transformer settings
    parser.add_argument("--t_embed_dim", default=10, type=int)
    parser.add_argument("--tf_head_num", default=2, type=int)
    parser.add_argument("--tf_layer_num", default=2, type=int)
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--bias", default=False, type=bool)

    # optimization settings
    parser.add_argument("--grad_clip", default=0, type=float) #5.0
    parser.add_argument("--eval_only", default=False, type=bool)
    parser.add_argument("--debug", default=False, type=bool)
    parser.add_argument("--eval_interval", default="1", type=int)
    parser.add_argument("--n_rgcn_layers_urban", default=1, type=int)
    parser.add_argument("--urban_emb_size", default=32, type=int)

    args = parser.parse_args()
    helpers.set_random_seed(args.seed)
    args.hostname = socket.gethostname()
    args.datapath = f"./data/{args.data}"
    device = torch.device(
        f"cuda:{args.cuda}" if torch.cuda.is_available() else torch.device("cpu")
    )

    #采用混合精度训练
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]
    args.ctx = torch.autocast(device_type="cuda", dtype=ptdtype)

    args.out_dir = args.out_dir + "/main"
    train_str = f"{args.seed}-{args.gnn}-gly{args.gnn_layer_num}-ghd{args.gnn_head_num}-tly{args.tf_layer_num}-thd{args.tf_head_num}"

    # set log path
    log_dir = f"./logs"
    log_prefix = (
        f"{args.method}-{args.data}-{train_str}-train-{args.hostname}-gpu{args.cuda}"
    )
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = helpers.set_logger(log_dir=log_dir, log_prefix=log_prefix)
    logger.info(args)

    # set saved path
    os.makedirs(args.out_dir, exist_ok=True)
    path_model_best = f"{args.out_dir}/{args.data}_{train_str}_ckpt_best.pth"
    path_model_last = f"{args.out_dir}/{args.data}_{train_str}_ckpt_last.pth"



    # load data
    adj_mx = helpers.read_adjcent_file(args.data)
    adj_dense = adj_mx.toarray()

    adj_no_isolate_file = f"./data/{args.data}/adjacent_mx_fill.npz_1.npy"
    if os.path.exists(adj_no_isolate_file):
        adj_dense = np.load(adj_no_isolate_file)
    else:
        for i in range(len(adj_dense)):
            adj_dense[i][i] = 1
            assert adj_dense[i].sum() > 0
            if adj_dense[i].sum() == 0:
                print(i)
                adj_dense[i][np.random.randint(0, len(adj_dense), 1)[0]] = 1
        np.save(adj_no_isolate_file, adj_dense)


    node_features, vocab_size = helpers.read_node_feature_file(args.data, device)
    args.vocab_size = vocab_size

    map_manager = helpers.MapManager(dataset_name=args.data)


    # use RKG
    base_graph_filename = "./data/{}/UrbanKG_data/base_knowledge_graph.csv".format(args.data)
    base_graph = pd.read_csv(base_graph_filename)
    base_graph = base_graph.values
    g_base = dgl.graph((base_graph[:, 0], base_graph[:, 2])).to(device)
    etype_base = torch.tensor(base_graph[:, 1].reshape(-1)).to(device)
    counts = torch.bincount(etype_base, minlength=4).cpu().detach().numpy()
    num_0=counts[0]
    num_1=counts[0]+counts[1]
    num_2=counts[0]+counts[1]+counts[2]
    adj_gra_mx=np.vstack((base_graph[:num_0,:], base_graph[num_2:,:]))
    adj_g_node = dgl.graph((adj_gra_mx[:,0],adj_gra_mx[:, 2])).to(device)
    adj_g_type = torch.tensor(adj_gra_mx[:, 1].reshape(-1)).to(device)
    no_adj_mx = np.vstack((base_graph[num_0:num_1, :], base_graph[num_1:num_2, :]))
    no_adj_g_node = dgl.graph((no_adj_mx[:, 0], no_adj_mx[:, 2])).to(device)
    no_adj_g_type = torch.tensor(no_adj_mx[:, 1].reshape(-1)).to(device)



    graph_data = {
        ('Road', 'Connectby', 'Road'): (base_graph[:num_0,0],base_graph[:num_0, 2]),
        ('Road', 'SpatialSimilarity', 'Road'): (base_graph[num_0:num_1,0],base_graph[num_0:num_1, 2]),
        ('Road', 'TemporalSimilarity', 'Road'): (base_graph[num_1:num_2,0],base_graph[num_1:num_2, 2]),
        ('Road', 'FlowTransition', 'Road'): (base_graph[num_2:,0],base_graph[num_2:, 2]),
    }
    hg_base = dgl.heterograph(graph_data).to(device)


    adj_graph, no_adj_graph, poi_distribution, time_distribution=helpers.proprecess_base_kg(args.data,hg_base,device)






    road_embed = 0  #use pretraing road
    road_embed = torch.tensor(road_embed).to(device)

    data_feature = {
        "adj_mx": adj_mx,
        "node_features": node_features,
        "img_width": map_manager.img_width,
        "img_height": map_manager.img_height,
    }
    gnn_config = {
        "gnn_model": args.gnn,
        "embed_dim": args.embed_dim,
        "no_gps_emb": True,
        "gps_emb_dim": args.gps_emb_dim,
        "num_of_layers": args.gnn_layer_num,
        "num_of_heads": args.gnn_head_num,
        "concat": False,
        "distance_mode": "l2",
        "hg_base":hg_base,
        "g_base":g_base,
        "etype_base":etype_base,
        "pretrain_road_embedded":road_embed,
        "adj_g_node":adj_g_node,
        "adj_g_type":adj_g_type,
        "no_adj_g_node": no_adj_g_node,
        "no_adj_g_type": no_adj_g_type,
        "poi_distribution": poi_distribution,
        "time_distribution": time_distribution,

    }
    tf_config = {
        "n_embd": args.embed_dim+ args.t_embed_dim,  # args.embed_dim includes node feature and gps_embd
        "t_embd": args.t_embed_dim,
        "block_size": map_manager.block_size,
        "n_head": args.tf_head_num,
        "n_layer": args.tf_layer_num,
        "dropout": args.dropout,
        "bias": args.bias,
    }


    #load  E-RKG
    augment_graph_filename="./data/{}/UrbanKG_data/aug_knowledge_graph.csv".format(args.data)
    augment_graph=pd.read_csv(augment_graph_filename)
    augment_graph=augment_graph.values
    g_urban = dgl.graph((augment_graph[:, 0], augment_graph[:, 2])).to(device)
    etype_urban = torch.tensor(augment_graph[:, 1].reshape(-1)).to(device)

    #load E-RKG embedding from GIE model
    entity_embedding_filename="./data/{}/UrbanKG_data/entity_new_kg_embedddings_gie_32.npy".format(args.data)
    entity_embedding_mx=np.load(entity_embedding_filename)
    entity_embedding_tensor = torch.tensor(entity_embedding_mx).to(torch.float32).to(device)

    #kgID to road_ID
    road = pd.read_csv(f"data/{args.data}/UrbanKG_data/kgID2roadID.csv")
    road_KG_id = road.values


    # load model
    model_args = dict(
        gnn_config=gnn_config,
        data_feature=data_feature,
        tf_config=tf_config,
        seed=args.seed,
        data=args.data,
        datapath=args.datapath,
        vocab_size=args.vocab_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        n_rgcn_layers_urban=args.n_rgcn_layers_urban,
        urban_emb_size=args.urban_emb_size,
    )
    model = MyTransformer(MyTransformerConfig(**model_args)).to(device)


    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr
    )

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="max",
        patience=args.lr_patience,
        factor=args.lr_decay_ratio,
    )

    # scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))

    logger.info(model)
    best_val_avg_acc_top1 = 0



    tra_loader, val_loader = helpers.generate_data_loader(
        args.data,
        "tra_and_val",
        args.batch_size,
        adj_dense,
        device,
    )




    for epoch in range(args.epochs):

        model.train()
        train_total_loss=0
        train_total_spatial_loss,val_total_spatial_loss = 0,0
        train_total_time_loss,val_total_time_loss = 0,0
        train_hit, train_cnt = 0, 0



        for batch in tqdm(tra_loader, desc="train transformer"):
            optimizer.zero_grad(set_to_none=True)
            logits_masked,_, loss,spatial_loss,time_loss = model(
                batch[0], batch[1], batch[2], batch[3], batch[4],
                g_urban,etype_urban,entity_embedding_tensor,road_KG_id,
                road_embed
            )

            loss.backward()
            train_total_loss += loss.item()
            train_total_spatial_loss += spatial_loss.item()
            train_total_time_loss += time_loss.item()

            if args.grad_clip != 0.0: #梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            value, index = torch.topk(logits_masked, 1, dim=-1)
            train_hit += (index.squeeze(-1) == batch[1]).sum()
            train_cnt += (batch[1] != -1).sum()
            if args.debug:
                break

        avg_acc = train_hit / train_cnt

        writer.add_scalar("acc", avg_acc, epoch)
        writer.add_scalar("loss", train_total_loss / len(tra_loader), epoch)
        writer.add_scalar("spatial_loss", train_total_spatial_loss / len(tra_loader), epoch)
        writer.add_scalar("temporal_loss", train_total_time_loss / len(tra_loader), epoch)

        logger.info(
            "epoch {}: train_loss {:.6f}, train_acc {:.6f}".format(epoch, train_total_loss / len(tra_loader),avg_acc))

        if (epoch + 1) % args.eval_interval == 0:
            model.eval()
            val_hit, val_cnt = 0, 0
            val_total_loss,val_total_spatial_loss,val_total_time_loss = 0,0,0
            start_i = 0
            for batch in tqdm(val_loader, desc="valid transformer"):
                with args.ctx:
                    logits_masked, _, loss, valid_spatial_loss, valid_temporal_loss = model(
                        batch[0], batch[1], batch[2], batch[3], batch[4],
                        g_urban,etype_urban,entity_embedding_tensor,road_KG_id,
                        road_embed
                    )
                val_total_loss += loss.item()
                val_total_spatial_loss += valid_spatial_loss.item()
                val_total_time_loss+= valid_temporal_loss.item()

                value, index = torch.topk(logits_masked, 1, dim=-1)
                val_hit += (index.squeeze(-1) == batch[1]).sum()
                val_cnt += (batch[1] != -1).sum()

            avg_ac = val_hit / val_cnt
            lr_scheduler.step(avg_ac)
            lr = optimizer.param_groups[0]['lr']
            writer2.add_scalar("acc", avg_ac, epoch)
            writer2.add_scalar("loss", val_total_loss / len(val_loader), epoch)
            writer2.add_scalar("valid_spatial_loss", val_total_spatial_loss / len(val_loader), epoch)
            writer2.add_scalar("valid_temporal_loss", val_total_time_loss / len(val_loader), epoch)
            logger.info("epoch {}: valid_loss {:.6f} val_acc {:.6f} ({} / {}),lr {}".format(epoch, val_total_loss / len(val_loader),avg_ac, val_hit, val_cnt,lr))

            if avg_ac > best_val_avg_acc_top1:
                best_val_avg_acc_top1 = avg_ac
                ckpt = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "epoch": epoch,
                    "best_val_avg_acc_top1": best_val_avg_acc_top1,
                }
                logger.info(f"saving checkpoint to {args.out_dir}")
                torch.save(ckpt, path_model_best)
            if lr < args.early_stop_lr:
                logger.info('early stop')
                break
            model.train()

    writer.close()
    writer2.close()

    # save model
    ckpt_last = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_args": model_args,
        "epoch": epoch,
        "best_val_avg_acc_top1": avg_ac,
    }
    torch.save(ckpt_last, path_model_last)
