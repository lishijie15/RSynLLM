import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy as cp
import math

class SpaceEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(SpaceEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe[:, :1177]

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(2)].unsqueeze(1)

class DataEmbedding_inverted(nn.Module):
    def __init__(self, seq_len, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x)
        # x: [Batch Variate d_model]
        return self.dropout(x)

class DataEmbedding_SE(nn.Module):
    def __init__(self, input_dim, d_model, seq_len, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_SE, self).__init__()
        self.spa_embedding = SpaceEmbedding(d_model=d_model)
        self.se_embedding = DataEmbedding_inverted(seq_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        x = self.se_embedding(x) + self.spa_embedding(x)
        return self.dropout(x)
    
    
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


class LoadExp(nn.Module):

    def __init__(self, args):
        super().__init__()
        # 初始化参数
        self.num_nodes = args.num_nodes
        self.d_model = args.d_model
        self.cheb_k = args.max_diffusion_step
        self.scales = [self.d_model // 6, self.d_model // 2, self.d_model]
        self.stride = args.patch_stride

        self.GCN = AGCN(self.d_model, self.d_model, self.cheb_k)

        self.scale_blocks = nn.ModuleDict({
            f'scale_{s}': nn.Sequential(
                DynamicPatching(scale=s, stride=self.stride, num_nodes=self.num_nodes),
                nn.Linear(s, s),
                Mixer4D(args=args,
                    input_dim=s,
                    node_dim=self.num_nodes,
                    channel_dim=self.d_model
                )
            ) for s in self.scales
        })

        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(args.d_model, len(self.scales)),
            nn.Softmax(dim=1)
        )

        self.channel_proj = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=args.d_model, kernel_size=1),  # 明确处理通道维度
            nn.BatchNorm2d(args.d_model),
            nn.GELU()
        )

    def reconstruct(self, patched, scale, stride, original_length):
        B, N, C, Num_patches, d = patched.shape
        patched = patched.permute(0, 1, 2, 4, 3).contiguous()  # [B,N,C,d,Num_patches]
        patched = patched.view(B * N * C, d, Num_patches)
        output = torch.nn.functional.fold(
            input=patched,
            output_size=(1, original_length),
            kernel_size=(1, scale),
            stride=(1, stride)
        )

        output = output.view(B, N, C, original_length)
        return output

    def forward(self, x, supports):
        x = self.GCN(x.permute(0, 2, 1, 3), supports).permute(0, 1, 3, 2)

        scale_features = []
        for s in self.scales:
            block = self.scale_blocks[f'scale_{s}']

            patched = block[0](x)

            embedded = block[1](patched)

            mixed = block[2](embedded)

            recon = self.reconstruct(
                patched=mixed,
                scale=s,
                stride=self.stride,
                original_length=self.d_model,
            )

            scale_features.append(recon)

        x = self.channel_proj(x.permute(0, 3, 2, 1))

        route_weights = self.router(x)  # [B, 3]

        fused = sum(
            w.view(-1, 1, 1, 1) * f
            for w, f in zip(route_weights.unbind(1), scale_features)
        )

        return fused


class DynamicPatching(nn.Module):

    def __init__(self, scale, stride, num_nodes):
        super().__init__()
        self.scale = scale
        self.stride = stride

    def forward(self, x):
        if x.size(-1) % self.scale != 0:
            pad = self.scale - (x.size(-1) % self.scale)
            x = F.pad(x, (0, pad))

        unfolded = x.unfold(-1, self.scale, self.stride)  # [B,N,C,Num_patches,Scale]
        return unfolded


class Mixer4D(nn.Module):

    def __init__(self, args, input_dim, node_dim, channel_dim):
        super().__init__()
        self.d_model = args.d_model
        self.node_mixer = nn.Sequential(
            nn.Linear(node_dim, node_dim * 2),
            nn.GELU(),
            nn.Linear(node_dim * 2, node_dim)
        )

        self.channel_mixer = nn.Sequential(
            nn.Linear(channel_dim, channel_dim * 2),
            nn.GELU(),
            nn.Linear(channel_dim * 2, channel_dim)
        )


        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.d_model, self.d_model, kernel_size=(1, 1),
                          padding=0, groups=self.d_model),
                nn.Conv2d(self.d_model, self.d_model, kernel_size=1)
            )
        ])

    def forward(self, x):
        B, N, C, Num_patches, d = x.shape

        multi_scale = x.reshape(B*N, C, Num_patches, d) + [conv(x.reshape(B*N, C, Num_patches, d)) for conv in self.conv_layers][0]
        multi_scale = multi_scale.reshape(B, N, C, Num_patches, d)

        return multi_scale


class EnhancedRouter(nn.Module):
    def __init__(self, d_model, che_k):
        super().__init__()

        self.spatial_conv = nn.Conv2d(
            in_channels=d_model,
            out_channels=d_model // 6,
            kernel_size=(1, 3),
            padding=(0, 1)
        )

        self.spatial_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.spatial_fc = nn.Conv2d(d_model // 6, 1, kernel_size=1)


        self.GCN = AGCN(d_model // 6, d_model // 6, che_k)

        self.temporal_route = nn.Sequential(
            nn.Conv1d(d_model, 16, kernel_size=3, padding=1),
            nn.Conv1d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.mode_fc = nn.Linear(d_model, 3)

    def forward(self, x, supports):
        x_ = self.spatial_conv(x.permute(0, 1, 3, 2))
        gcn_output = self.GCN(x_.permute(0, 3, 2, 1), supports)
        s = self.spatial_pool(gcn_output.permute(0, 3, 1, 2))
        s_weight =torch.sigmoid(self.spatial_fc(s))


        t_weight = self.temporal_route(x.mean(dim=2))
        t_weight = t_weight.unsqueeze(2)

        mode_weight = F.softmax(self.mode_fc(x.mean(dim=(2, 3))), dim=1)

        fused_weight = s_weight * t_weight

        return (fused_weight.unsqueeze(1) * mode_weight.view(-1, 3, 1, 1, 1)).squeeze(2)

class ThermalExp(nn.Module):
    def __init__(self, args):
        super(ThermalExp, self).__init__()
        self.channel_in = args.input_dim
        self.d_model = args.d_model
        self.cheb_k = args.max_diffusion_step
        self.lag_window = 24
        self.GCN = AGCN(self.d_model, self.d_model, self.cheb_k)

        self.pad_layer, self.lag_conv = self._build_lag_conv()

        self.route_controller = EnhancedRouter(self.d_model, self.cheb_k)

        self.mode_branches = nn.ModuleList([
            nn.Identity(),
            AGCN(self.d_model, self.d_model, self.cheb_k),
            nn.Conv1d(self.d_model, self.d_model, 3, padding=1)
        ])

    def _build_lag_conv(self):
        if self.lag_window % 2 == 0:
            pad_left = (self.lag_window // 2) - 1
            pad_right = self.lag_window // 2
            pad_layer = nn.ConstantPad1d((pad_left, pad_right), 0)
        else:
            pad_total = (self.lag_window - 1) // 2
            pad_layer = nn.ConstantPad1d(pad_total, 0)

        lag_conv = nn.Conv1d(
            self.d_model, self.d_model,
            kernel_size=self.lag_window,
            groups=self.d_model
        )
        return pad_layer, lag_conv


    def forward(self, x, supports):
        x = self.GCN(x.permute(0, 2, 3, 1), supports).permute(0,3,1,2) # 变成B N T C

        B, D, N, T = x.shape
        x_merged = x.permute(0, 2, 1, 3).reshape(B * N, D, T)
        x_lag = self.lag_conv(self.pad_layer(x_merged))
        x_lag = x_lag.view(B, N, D, T).permute(0, 2, 1, 3)  # [B,D,N,T]

        route_weights = self.route_controller(x_lag, supports)

        mode_outputs = []
        for i, branch in enumerate(self.mode_branches):
            if i == 0:
                mode_out = x_lag
            elif i == 1:
                mode_out = branch(x_lag.permute(0, 2, 3, 1), supports).permute(0, 3, 1, 2)
            else:
                x_lag = x_lag.permute(0, 2, 1, 3).reshape(B * N, D, T)
                mode_out = branch(x_lag).reshape(B, N, D, T).permute(0, 2, 1, 3)
            mode_outputs.append(mode_out * route_weights[:, i].unsqueeze(1))

        final_out = sum(mode_outputs)
        return final_out


class GasExp(nn.Module):
    def __init__(self, args):
        super(GasExp, self).__init__()
        self.num_nodes = args.num_nodes
        self.d_model = args.d_model
        self.channel_in = args.input_dim

        self.gate_gen = nn.Sequential(
            nn.Conv2d(self.d_model, 32, kernel_size=(1, 3), padding=(0, 1)),
            nn.GELU(),
            nn.Conv2d(32, 2, kernel_size=1)
        )

        self.res_conv = nn.Sequential(
            nn.ConstantPad1d((3, 4), 0),
            nn.Conv1d(
                in_channels=self.d_model,
                out_channels=self.d_model,
                kernel_size=8,
                padding=0,
                groups=self.d_model
            ),
            nn.GELU()
        )


        self.ind_conv = nn.Sequential(
            nn.ConstantPad1d((83, 84), 0),
            nn.Conv1d(
                in_channels=self.d_model,
                out_channels=self.d_model,
                kernel_size=24 * 7,
                padding=0,  # (24 * 7) // 2
                groups=self.d_model
            ),
            nn.GELU()
        )

    def forward(self, x):
        # x: [B, C, N, T], node_feat: [N, D]
        B, C, N, T = x.shape

        x_res = x.permute(0, 2, 1, 3)  # [B, N, C, T]
        x_res = x_res.reshape(B * N, C, T)  # [B*N, C, T]
        res_feat = self.res_conv(x_res)  # [B*N, D, T]
        res_feat = res_feat.view(B, N, self.d_model, T)  # [B, N, D, T]
        res_feat = res_feat.permute(0, 2, 1, 3)  # [B, D, N, T]

        x_ind = x.permute(0, 2, 1, 3)  # [B, N, C, T]
        x_ind = x_ind.reshape(B * N, C, T)
        ind_feat = self.ind_conv(x_ind)  # [B*N, D, T]
        ind_feat = ind_feat.view(B, N, self.d_model, T)
        ind_feat = ind_feat.permute(0, 2, 1, 3)  # [B, D, N, T]

        gate = torch.softmax(self.gate_gen(x), dim=1)

        return gate[:, 0:1] * res_feat + gate[:, 1:2] * ind_feat  # [B, D, N, T]


class BatchNorm(nn.Module):
    def __init__(self, num_features, momentum=0.1, eps=1e-5, track_running_stats=True):
        super(BatchNorm, self).__init__()
        self.momentum = momentum
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        if track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)

    def forward(self, x):
        dims = [i for i in range(x.dim() - 1)]
        mean = x.mean(dim=dims)
        var = x.var(dim=dims, correction=0)
        if (self.training) and (self.running_mean is not None):
            avg_factor = self.momentum
            moving_avg = lambda prev, cur: (1 - avg_factor) * prev + avg_factor * cur.detach()
            dims = [i for i in range(x.dim() - 1)]
            self.running_mean = moving_avg(self.running_mean, mean)
            self.running_var = moving_avg(self.running_var, var)
            mean, var = self.running_mean, self.running_var

        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        out = x_norm * self.gamma + self.beta
        return out