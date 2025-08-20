import math
from dataclasses import dataclass
from typing import Dict
from dgl.nn.pytorch import GraphConv,RelGraphConv
import helpers
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from distance_gat_fc import DistanceGatFC
from torch.nn import functional as F
from torch.profiler import ProfilerActivity, profile, record_function
from tqdm import tqdm
import inspect
from math import sqrt

class TimeEncode(torch.nn.Module):
    def __init__(self, expand_dim):
        super(TimeEncode, self).__init__()
        time_dim = expand_dim    # time_dim 为   10
        self.basis_freq = torch.nn.Parameter(
            (torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim))).float()
        )
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float())

    def forward(self, ts):
        # ts: [N, L]
        batch_size = ts.size(0)
        seq_len = ts.size(1)

        ts = ts.view(batch_size, seq_len, 1)  # [N, L, 1]
        map_ts = ts * self.basis_freq.view(1, 1, -1)  # [N, L, time_dim]
        map_ts += self.phase.view(1, 1, -1)

        harmonic = torch.cos(map_ts)
        return harmonic  # self.dense(harmonic)


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

'''
      使用因果注意力机制进行融合  
'''

class CausalAttention2Infusion(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v,config):
        super(CausalAttention2Infusion, self).__init__( )
        self.dim_q = dim_q
        self.dim_k = dim_k
        self.dim_v = dim_v

        # 定义线性变换函数
        self.linear_q = nn.Linear(dim_q, dim_k, bias = False)
        self.linear_k = nn.Linear(dim_q, dim_k, bias = False)
        self.linear_v = nn.Linear(dim_q, dim_v, bias = False)
        self._norm_fact = 1 / torch.sqrt(torch.tensor(dim_k, dtype=torch.float32))   #缩放点积注意力机制

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config["block_size"], config["block_size"])).view(
                 1,1, config["block_size"], config["block_size"]
            ),
        )


        self.register_buffer(
            "k_cache",
            torch.empty(
                config["batch_size"],
                config["block_size"],
                config["n_embd"],
            ),
            persistent=False,
        )

        self.register_buffer(
            "v_cache",
            torch.empty(
                config["batch_size"],
                config["block_size"],
                config["n_embd"],
            ),
            persistent=False,
        )

    def infer_next(self, start_pos: int, x: torch.Tensor):
        # x: batch, n, dim_q

        batch, n, dim_q = x.shape
        assert dim_q == self.dim_q

        q = self.linear_q(x)  # batch, n, dim_k

        k = self.linear_k(x)  # batch, n, dim_k

        v = self.linear_v(x)  # batch, n, dim_v

        kc = self.get_buffer("k_cache")
        vc = self.get_buffer("v_cache")
        kc[:batch, start_pos: start_pos + n, :] = k
        vc[:batch, start_pos: start_pos + n, :] = v

        k: torch.Tensor = (kc[:batch, : start_pos + n])
        v: torch.Tensor = (vc[:batch, : start_pos + n])



        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact  # batch, n, n
        dist = dist.masked_fill(self.bias[:, start_pos:start_pos + n, :start_pos + n] == 0, float("-inf"))
        # dist = dist.masked_fill(mask == 0, float('-1e20'))  # 将掩码为0的位置设置为负无穷

        dist = torch.softmax(dist, dim=-1)  # batch, n, n

        att = torch.bmm(dist, v)  # batch, n, dim_v

        return att


    def forward(self, x):
        # x: batch, n, dim_q

        batch, n, dim_q = x.shape
        assert dim_q == self.dim_q

        q = self.linear_q(x)  # batch, n, dim_k

        k = self.linear_k(x)  # batch, n, dim_k

        v = self.linear_v(x)  # batch, n, dim_v


        mask = torch.tril(torch.ones(n, n, device=x.device), diagonal=0)
        mask = mask.unsqueeze(0).repeat(batch, 1, 1)  # batch, n, n


        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact  # batch, n, n
        dist = dist.masked_fill(mask == 0, float('-1e20'))


        dist = torch.softmax(dist, dim=-1)  # batch, n, n


        att = torch.bmm(dist, v)  # batch, n, dim_v

        return att


class Linear2Infusion(nn.Module):
    def __init__(self, config):
        super(Linear2Infusion, self).__init__( )
        self.linear=nn.Linear(config["n_embd"],config["n_embd"])
        self.linear_KG=nn.Linear(config["n_embd"],config["n_embd"])
        self.gelu=nn.GELU()

    def forward(self,road_embedding,entity_embedding):
        road_embedding_=self.linear(road_embedding)
        entity_embedding_=self.linear_KG(entity_embedding)
        out_put=self.gelu(road_embedding_+entity_embedding_)

        return out_put



class Linear2Infusion2(nn.Module):
    def __init__(self, config):
        super(Linear2Infusion2, self).__init__( )
        self.linear=nn.Linear(config["n_embd"]*2,config["n_embd"])


    def forward(self,road_embedding):
        road_embedding_=self.linear(road_embedding)
        out_put=self.gelu(road_embedding_)

        return out_put



class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config["n_embd"] % config["n_head"] == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(
            config["n_embd"], 3 * config["n_embd"], bias=config["bias"]
        )
        # output projection
        self.c_proj = nn.Linear(config["n_embd"], config["n_embd"], bias=config["bias"])
        self.attn_dropout = nn.Dropout(config["dropout"])
        self.resid_dropout = nn.Dropout(config["dropout"])
        self.n_head = config["n_head"]
        self.n_embd = config["n_embd"]
        self.dropout = config["dropout"]
        # support only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        self.flash = False
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config["block_size"], config["block_size"])).view(
                    1, 1, config["block_size"], config["block_size"]
                ),
            )

        self.register_buffer(
            "k_cache",
            torch.empty(
                config["batch_size"],
                config["block_size"],
                config["n_embd"],
            ),
            persistent=False,
        )

        self.register_buffer(
            "v_cache",
            torch.empty(
                config["batch_size"],
                config["block_size"],
                config["n_embd"],
            ),
            persistent=False,
        )

    @torch.no_grad()
    def infer_next(self, start_pos: int, x: torch.Tensor):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        kc = self.get_buffer("k_cache")
        vc = self.get_buffer("v_cache")

        kc[:B, start_pos: start_pos + T, :] = k
        vc[:B, start_pos: start_pos + T, :] = v

        q: torch.Tensor = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k: torch.Tensor = (
            kc[:B, : start_pos + T]
                .view(B, -1, self.n_head, C // self.n_head)
                .transpose(1, 2)
        )
        v: torch.Tensor = (
            vc[:B, : start_pos + T]
                .view(B, -1, self.n_head, C // self.n_head)
                .transpose(1, 2)
        )

        assert not self.flash
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        assert T == 1
        if T != 1:
            att = att.masked_fill(self.bias[:, :, start_pos:start_pos + T, :start_pos + T] == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            # att_numpy=att.cpu().detach().numpy()
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(
            config["n_embd"], 4 * config["n_embd"], bias=config["bias"]
        )
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(
            4 * config["n_embd"], config["n_embd"], bias=config["bias"]
        )
        self.dropout = nn.Dropout(config["dropout"])

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config["n_embd"], bias=config["bias"])
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config["n_embd"], bias=config["bias"])
        self.mlp = MLP(config)

    @torch.no_grad()
    def infer_next(self, start_pos: int, x: torch.Tensor):
        x = x + self.attn.infer_next(start_pos, self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class MyTransformerConfig:
    gnn_config: Dict
    data_feature: Dict
    tf_config: Dict
    seed: int = 0
    data: str = ""
    datapath: str = ""
    vocab_size: int = 0
    epochs: int = 50
    batch_size: int = 32
    device: torch.device = torch.device("cuda:0")

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


class MyTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.tf_config["block_size"] is not None
        self.config = config
        self.device = config.device
        self.loss = nn.MSELoss(reduction="none")

        config.tf_config["batch_size"] = config.batch_size

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(
                    config.vocab_size,
                    config.tf_config["n_embd"] - config.tf_config["t_embd"],
                ),
                tme=TimeEncode(config.tf_config["t_embd"]),
                wpe=nn.Embedding(
                    config.tf_config["block_size"], config.tf_config["n_embd"]
                ),
                drop=nn.Dropout(config.tf_config["dropout"]),
                h=nn.ModuleList([Block(config.tf_config) for _ in range(config.tf_config["n_layer"])]),
                ln_f=LayerNorm(
                    config.tf_config["n_embd"], bias=config.tf_config["bias"]
                ),
            )
        )
        self.lm_head = nn.Linear(
            config.tf_config["n_embd"], config.vocab_size, bias=False
        )
        self.time_pred = nn.Linear(config.tf_config["n_embd"], 1, bias=False)


        self.gat = DistanceGatFC(
            config=config.gnn_config,
            data_feature=config.data_feature,
            device=config.device,
        )

        ##R-Gcn
        self.rgcns_urban = nn.ModuleList()
        for i in range(config.n_rgcn_layers_urban):
            self.rgcns_urban.append(RelGraphConv(config.urban_emb_size, config.tf_config["n_embd"], 17, 'basis', num_bases=4))



        #linear fusion
        self.linear2fusion=Linear2Infusion(config.tf_config)

        # CausalAttention   fusion
        # self.attention = CausalAttention2Infusion(config.tf_config["n_embd"] + config.tf_config["n_embd"], config.tf_config["n_embd"], config.tf_config["n_embd"],config.tf_config)


        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.tf_config["n_layer"])
                )
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_acc_topk(self, preds, targets):
        acc_K = [1, 5, 10, 20]
        result = {}
        totalMRR = []
        for K in acc_K:
            result[K] = 0

        seq_len_l = []
        for i in range(len(preds)):
            max_len = self.config.tf_config["block_size"] - 1
            seq_len = max_len - len(torch.where(targets[i] == -1)[0])
            seq_len_l.append(seq_len)

            for j in range(seq_len):
                pred, target = preds[i][j], targets[i][j].item()
                sortedPred = torch.topk(pred, len(pred))[1].tolist()
                truthIndex = sortedPred.index(target) + 1
                avgPrec = 1 / truthIndex
                totalMRR.append(avgPrec)

                sorted_indexs = {}
                for K in acc_K:
                    sorted_indexs[K] = sortedPred[:K]
                    if target in sorted_indexs[K]:
                        result[K] += 1

        result["num_of_test"] = sum(seq_len_l)
        result["mrr"] = np.sum(totalMRR)
        result["mrr_num"] = len(totalMRR)
        return result


    @torch.no_grad()
    def infer_next(
            self,
            start_pos: int,
            idx,
            tim_real,
            adj_batch,
            gnn_emb,g_urban,etype_urban,entity_embedding_tensor,road_KG_id,road_embed):

        assert gnn_emb is not None
        tok_emb = gnn_emb[idx]
        tim_emb = self.transformer.tme(tim_real)  # [batch, 1, t_embd]

        device = idx.device
        T = idx.size(1)
        assert (
                start_pos + T <= self.config.tf_config["block_size"]
        ), f"Cannot forward sequence of length {start_pos + T}, block size is only {self.config.tf_config['block_size']}"
        pos = torch.arange(
            start_pos, start_pos + T, dtype=torch.long, device=device
        ).unsqueeze(0)
        pos_emb = self.transformer.wpe(pos)  # [1, len-1, n_embd + t_embd]

        x = self.transformer.drop(torch.concat((tok_emb, tim_emb), dim=-1) + pos_emb)
        for block in self.transformer.h:
            x = block.infer_next(start_pos, x)

        x = self.transformer.ln_f(x)

        '''
               知识图谱引导模块
        '''
        for i in range(self.n_rgcn_layers_urban):  # 32
            entity_embedding_tensor = self.rgcns_urban[i](g_urban, entity_embedding_tensor, etype_urban)

        idx_np = idx.cpu().numpy()
        kg_ids = road_KG_id[idx_np]
        kg_ids_squeezed = np.squeeze(kg_ids, axis=2)
        kg_ids_tensor = torch.tensor(kg_ids_squeezed).to(self.device)
        env_emb = entity_embedding_tensor[kg_ids_tensor]
        x = self.linear2fusion(x, env_emb)


        #pred next road
        logits_weighted = (
                self.lm_head(x[:, [-1], :])
        )  # (batch, 1, n)
        logits_masked = torch.where(
            adj_batch[:, [-1], :] == 0,
            torch.tensor(float("-inf")).to(self.device),
            logits_weighted,
        )  # (batch, 1, n)

        # pred next duration time
        dur_pred = self.time_pred(x[:, [-1], :]).squeeze(-1)  # (batch, 1)
        dur_pred = torch.clamp(dur_pred, min=0.0)

        loss = None

        return logits_masked, dur_pred, loss

    def forward(
            self,
            idx,
            targets=None,
            tim_real=None,
            adj_batch=None,
            dur_time_batch=None,
            g_urban=None,
            etype_urban=None,
            entity_embedding_tensor=None,
            road_KG_id=None,
            road_embed=None,
            gnn_emb=None,
    ):
        """
        idx (input): (batch, len-1)
        targets (output): (batch, len-1)
        dest (destination): (batch, len-1)
        tim_real:
        adj_batch: (batch, len-1, n)
        dist_geo_batch: (batch, len-1, n)
        """


        if gnn_emb == None:
            assert targets != None
            gnn_emb = self.gat.compute_gnn()  # [loc_num, n_embd]

        tok_emb = gnn_emb[idx]  # [batch, len-1, n_embd]



        assert targets is not None
        if targets is not None:
            tim_emb = self.transformer.tme(tim_real[:, :-1])  # [batch, len-1, t_embd]
        else:
            tim_emb = self.transformer.tme(tim_real)  # [batch, len-1, t_embd]

        device = idx.device
        t = idx.size()[1]
        assert (
                t <= self.config.tf_config["block_size"]
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.tf_config['block_size']}"


        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(torch.concat((tok_emb, tim_emb), dim=-1) + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        for i in range(self.n_rgcn_layers_urban):  #32
            entity_embedding_tensor = self.rgcns_urban[i](g_urban,entity_embedding_tensor,etype_urban)


        idx_np = idx.cpu().numpy()
        kg_ids = road_KG_id[idx_np]
        kg_ids_squeezed = np.squeeze(kg_ids, axis=2)
        kg_ids_tensor = torch.tensor(kg_ids_squeezed).to(self.device)
        env_emb=entity_embedding_tensor[kg_ids_tensor]

        x=self.linear2fusion(x,env_emb) #fusion

        import torch.utils.checkpoint as ckpt_utils

        def geo_func(x):
            # N, L, C = dist_geo_batch_x.shape
            len_mask = tim_real[:, 1:] == -1  # (batch, len-1) 填充值为true 非填充值为false

            # pretraining, target is the correct next step rid
            assert targets is not None
            if targets is not None:
                logits_weighted = (
                        self.lm_head(x)
                )  # distance activation (batch, len-1, n)
                logits_masked = torch.where(
                    adj_batch == 0,
                    torch.tensor(float("-inf")).to(self.device),
                    logits_weighted,
                )  # adj mask (batch, len-1, n)
                spatial_loss = F.cross_entropy(
                    logits_masked.view(-1, logits_masked.size(-1)),
                    targets.reshape(-1),
                    ignore_index=-1,
                )


                # duration prediction
                dur_pred = self.time_pred(x).squeeze(-1)  # (batch, len-1)
                dur_real = dur_time_batch.clone().detach().type(torch.float32)

                dur_real[len_mask] = 0
                dur_pred[len_mask] = 0

                loss_dur = self.loss(dur_pred, dur_real)  # (batch, len-1)
                loss_dur_masked = (
                        loss_dur * (~len_mask).float()
                ).sum()  # gives \sigma_euclidean over unmasked elements
                time_loss= loss_dur_masked / (~len_mask).sum()

                loss = spatial_loss + time_loss

            # generate seq based on last idx
            else:
                logits_weighted = (
                        self.lm_head(x[:, [-1], :]) * weight_dis[:, [-1], :]
                )  # (batch, 1, n)
                logits_masked = torch.where(
                    adj_batch[:, [-1], :] == 0,
                    torch.tensor(float("-inf")).to(self.device),
                    logits_weighted,
                )  # (batch, 1, n)

                # duration prediction
                dur_hat = self.time_pred(x[:, [-1], :]).squeeze(-1)  # (batch, 1)
                probs = logits_masked.softmax(dim=-1)
                prob_mask = torch.where(
                    adj_batch[:, [-1], :] == 0,
                    torch.tensor(float(0.0)).to(self.device),
                    probs,
                )
                delta_dis = (probs * weight_tim[:, [-1], :]).sum(
                    dim=-1
                )  # (batch, 1)
                dur_pred = dur_hat * delta_dis  # (batch, 1)
                loss = None
                spatial_loss = None
                time_loss = None
            return logits_masked, dur_pred, loss, spatial_loss, time_loss

        return ckpt_utils.checkpoint(geo_func, x)



    @torch.no_grad()
    def generate(self, args, adj, num_samples,g_urban,etype_urban,entity_embedding_tensor,road_KG_id,len_list,road_embed,temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """

        od_and_probs_float = helpers.read_od_pair_distribution(
            args.data
        )  # tensor(n, 3)
        od_ids = torch.multinomial(
            od_and_probs_float[:, -1], 5000, replacement=True
        )

        origins = (
            od_and_probs_float[od_ids, 0].int().reshape(-1, 1).to(self.device).long()
        )
        destinations = (
            od_and_probs_float[od_ids, 1].int().reshape(-1, 1).to(self.device).long()
        )

        start_t_probs = helpers.read_start_t_probs(args.data)
        assert len(start_t_probs) == 2880
        start_ts = (
            torch.multinomial(start_t_probs, num_samples, replacement=True)
            .reshape(-1, 1)
            .to(self.device)
        )


        len_vals, len_cnts = np.unique(len_list, return_counts=True)
        len_cnts = len_cnts / np.sum(len_cnts)
        samples_len = [np.random.choice(len_vals, 1, p = len_cnts)[0] for i in range(num_samples)]
        samples_len=np.array(samples_len).reshape(len(origins), 1)


        gen_seq_len = self.config.tf_config["block_size"] - 1


        gen_seq_batch = 32
        gnn_emb = self.gat.compute_gnn()
        pred_traj, pred_tim, pred = [], [], []
        batch_num = int(np.ceil(num_samples / gen_seq_batch))

        adj_tensor = torch.from_numpy(adj)
        for i in tqdm(range(batch_num)):
            sid_bch = i * gen_seq_batch
            eid_bch = min(sid_bch + gen_seq_batch, num_samples)

            bs = eid_bch - sid_bch
            idx_cond = torch.zeros(bs, gen_seq_len+1).type_as(origins)
            tim = torch.zeros(bs, gen_seq_len+1, dtype=torch.float32).to(self.device)

            idx_cond[:,0:1] = origins[sid_bch:eid_bch]
            tim[:,0:1] = start_ts[sid_bch:eid_bch]

            adj_batch = torch.zeros(bs, gen_seq_len+1, *adj_tensor.shape[1:],
                                    dtype=adj_tensor.dtype, device=self.device)
            adj_batch[:,0,:].copy_(adj_tensor[idx_cond[:,0].cpu()])

            for t in range(gen_seq_len):
                logits, dur_pred, _ = self.infer_next(
                    start_pos=t,
                    idx=idx_cond[:,t:t+1],
                    tim_real=torch.round(tim[:, t:t+1]).long(),
                    adj_batch=adj_batch[:, :t+1, :],
                    gnn_emb=gnn_emb,
                    g_urban=g_urban,
                    etype_urban=etype_urban,
                    entity_embedding_tensor=entity_embedding_tensor,
                    road_KG_id=road_KG_id,
                    road_embed=road_embed
                )  # (batch, 1, n)

                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")
                probs = F.softmax(logits, dim=-1)
                fill_matrix = torch.full_like(
                    probs, 1 / self.config.tf_config["block_size"]
                )
                probs = torch.where(torch.isnan(probs), fill_matrix, probs)
                idx_next = torch.multinomial(probs, num_samples=1)

                idx_next = idx_next.view(-1)
                idx_cond[:,t+1].copy_(idx_next)

                tmp = tim[:, t].clone()
                mask = (tmp >= 1440)
                tmp[mask] -= 1440
                tmp = (tmp + dur_pred.view(-1)) % 1440 #
                tmp[mask] += 1440

                tim[:,t+1].copy_(tmp)

                adj_batch[:,t+1].copy_(adj_tensor[idx_next.cpu()])

            pred_traj.extend(idx_cond.tolist())
            pred_tim.extend(torch.round(tim).long().tolist())

        for i in range(num_samples):
            pred_traj[i] = pred_traj[i][0:samples_len[i][0]]
            pred_tim[i] = pred_tim[i][0:samples_len[i][0]]


        if args.data=="Porto_Taxi":
            df = pd.read_csv(f"./data/{args.data}/porto.rel")
        elif args.data=="BJ_Taxi":
            df = pd.read_csv(f"./data/{args.data}/roadmap.rel")

        oid = df["origin_id"].tolist()
        did = df["destination_id"].tolist()
        stop_points = list(set(did) - set(oid))
        destinations = destinations.reshape(-1).cpu().tolist()
        map_manager = helpers.MapManager(args.data)
        to_des_cnt = 0
        to_max_traj_len=0
        s=0
        for i in range(len(pred_traj)):
            include_stop = len(set(pred_traj[i]).intersection(stop_points)) != 0
            if destinations[i] in pred_traj[i] and (   #
                not include_stop
            ):
                to_des_cnt += 1
                dest_pos = pred_traj[i].index(destinations[i])
                pred.append(
                    [
                        pred_traj[i][: dest_pos + 1],
                        pred_tim[i][: dest_pos + 1],
                        destinations[i],
                    ]
                )
                s+=1
            # 不会执行
            elif destinations[i] in pred_traj[i] and include_stop:
                stp = list(set(pred_traj[i]).intersection(stop_points))[0]
                stop_pos = pred_traj[i].index(stp)
                dest_pos = pred_traj[i].index(destinations[i])
                if (
                    dest_pos <= stop_pos
                ):
                    to_des_cnt += 1
                    pred.append(
                        [
                            pred_traj[i][: dest_pos + 1],
                            pred_tim[i][: dest_pos + 1],
                            destinations[i],
                        ]
                    )
                    s += 1
                elif stop_pos + 1 >= map_manager.min_len:
                    pred.append(
                        [
                            pred_traj[i][: stop_pos + 1],
                            pred_tim[i][: stop_pos + 1],
                            destinations[i],
                        ]
                    )
                    s += 1
            elif (
                not include_stop
            ):  # Known: des not in traj
                to_max_traj_len+=1
                length = samples_len[i][0]
                pred.append(
                    [pred_traj[i][:length], pred_tim[i][:length], destinations[i]]
                )
            else:
                stp = list(set(pred_traj[i]).intersection(stop_points))[0]
                stop_pos = pred_traj[i].index(stp)
                length = samples_len[i][0]
                if stop_pos + 1 >= map_manager.min_len and stop_pos + 1 <= length:
                    pred.append(
                        [
                            pred_traj[i][: stop_pos + 1],
                            pred_tim[i][: stop_pos + 1],
                            destinations[i],
                        ]
                    )
                elif (stop_pos + 1 >= map_manager.min_len and stop_pos + 1 > length):
                    pred.append(
                        [pred_traj[i][:length], pred_tim[i][:length], destinations[i]]
                    )


        return pred
