import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from RSynLLM.model.st_layers.routingMoE import DataEmbedding_SE, LoadExp, ThermalExp, GasExp
from transformers.configuration_utils import PretrainedConfig


class AGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, dropout=0.1):
        super(AGCN, self).__init__()
        self.cheb_k = cheb_k
        self.dropout = dropout
        self.MLP = nn.Linear((2 * cheb_k + 1) * dim_in, dim_out)

    def forward(self, x, supports): #B, N, T, C
        x_g = [x]
        support_set = []
        for support in supports:
            support_ks = [torch.eye(support.shape[0]).to(support.device), support]
            for k in range(2, self.cheb_k):
                support_ks.append(torch.matmul(2 * support, support_ks[-1]) - support_ks[-2])
            support_set.extend(support_ks)
        for support in support_set:
            x_g.append(torch.einsum("bntc,nm->bmtc", x, support))
        x_g = torch.cat(x_g, dim=-1)  # B, N, 2 * cheb_k * dim_in
        x_gconv = self.MLP(x_g)  # b, N, dim_out
        return x_gconv


class GCTblock_enc(nn.Module):
    def __init__(self, args):
        super(GCTblock_enc, self).__init__()
        self.input_window = args.input_window
        self.output_window = args.output_window
        self.num_nodes = args.num_nodes
        self.input_dim = args.input_dim
        self.output_dim = args.output_dim
        self.cheb_k = args.max_diffusion_step
        self.b_layers = args.b_layers
        self.d_model = args.d_model
        self.AGCNs = AGCN(self.d_model, self.d_model, self.cheb_k)

        self.layer_norm = nn.LayerNorm(self.d_model)
        self.data_embedding = DataEmbedding_SE(self.input_dim, self.d_model, self.input_window)
        self.LoadExp = LoadExp(args)
        self.ThermalExp = ThermalExp(args)
        self.GasExp = GasExp(args)
        self.single_load = nn.Linear(1, self.d_model)
        self.single_thermal = nn.Linear(1, self.d_model)
        self.single_gas = nn.Linear(1, self.d_model)
        self.total = nn.Linear(self.input_dim, self.d_model)

        self.router = nn.Sequential(
            nn.Conv2d(self.d_model, 32, kernel_size=(1, 3), padding=(0, 1)),
            nn.GELU(),
            nn.Conv2d(32, self.input_dim, kernel_size=1),
            nn.Softmax(dim=1)
        )
        self.chan_pred = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.output_dim)
        )

        # parameter-efficient design

    def forward(self, x, supports):
        # shape of x: (B, T, N, D)
        assert x.shape[2] == self.num_nodes and x.shape[3] == self.input_dim
        # embedding
        embed_inputs = self.data_embedding(x)  # [B, C, N, T]
        current_inputs = self.total(embed_inputs.permute(0, 2, 3, 1))
        shared_feat = self.AGCNs(current_inputs, supports)

        load, gas, heat = torch.chunk(embed_inputs, embed_inputs.size()[1], dim=1)
        load_emb = self.single_load(load.permute(0, 3, 2, 1)) #[B, T, N, C]
        gas_emb = self.single_gas(gas.permute(0, 3, 2, 1))  # [B, T, N, C]
        thermal_emb = self.single_thermal(heat.permute(0, 3, 2, 1))  # [B, T, N, C]
        
        load_out = self.LoadExp(load_emb, supports)
        gas_out = self.GasExp(gas_emb)
        thermal_out = self.ThermalExp(thermal_emb, supports)
        
        router_weights = self.router(shared_feat.permute(0, 3, 1, 2))
        fused_output = (router_weights[:, 0:1] * load_out.permute(0, 3, 1, 2) +
                        router_weights[:, 1:2] * gas_out +
                        router_weights[:, 2:3] * thermal_out)

        output = self.chan_pred(fused_output.permute(0, 3, 2, 1))
        return output

class ST_Enc(nn.Module):
    def __init__(self, args):
        super(ST_Enc, self).__init__()
        self.config = PretrainedConfig()
        self.num_nodes = args.num_nodes
        self.input_dim = args.input_dim
        self.input_window = args.input_window
        self.b_layers = args.b_layers
        self.d_model = args.d_model
        self.output_dim = args.output_dim
        self.output_window = args.output_window
        self.cheb_k = args.max_diffusion_step

        # memory
        self.mem_num = args.mem_num
        self.mem_dim = args.mem_dim
        self.memory = self.construct_memory()

        # encoder
        self.GCT_enc = GCTblock_enc(args)

        self.decoder_dim = self.d_model + self.mem_dim

        self.proj = nn.Sequential(nn.Linear(self.d_model, self.output_window, bias=True))

    def construct_memory(self):
        memory_dict = nn.ParameterDict()
        memory_dict['Memory'] = nn.Parameter(torch.randn(self.mem_num, self.mem_dim), requires_grad=True)  # (M, d)
        memory_dict['Wq'] = nn.Parameter(torch.randn(self.d_model, self.mem_dim),
                                         requires_grad=True)  # project to query
        memory_dict['We1'] = nn.Parameter(torch.randn(self.num_nodes, self.mem_num),
                                          requires_grad=True)  # project memory to embedding
        memory_dict['We2'] = nn.Parameter(torch.randn(self.num_nodes, self.mem_num),
                                          requires_grad=True)  # project memory to embedding
        for param in memory_dict.values():
            nn.init.xavier_normal_(param)
        return memory_dict

    def forward(self, x, labels=None, batches_seen=None):
        # mem_num：number of prototypes
        node_embeddings1 = torch.matmul(self.memory['We1'], self.memory['Memory'])
        node_embeddings2 = torch.matmul(self.memory['We2'], self.memory['Memory'])
        g1 = F.softmax(F.relu(torch.mm(node_embeddings1, node_embeddings2.T)), dim=-1)  # E,ET
        g2 = F.softmax(F.relu(torch.mm(node_embeddings2, node_embeddings1.T)), dim=-1)
        supports = [g1, g2]

        out_exp = self.GCT_enc(x, supports)

        output = self.proj(out_exp.permute(0, 3, 2, 1))

        return output.permute(0, 3, 2, 1), out_exp