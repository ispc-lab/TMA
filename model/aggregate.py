import torch
import torch.nn as nn
import torch.nn.functional as F


class MotionFeatureEncoder(nn.Module):
    def __init__(self, corr_level, corr_radius):
        super(MotionFeatureEncoder, self).__init__()
        cor_planes = corr_level * (2*corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+192, 128-2, 3, padding=1)

    def forward(self, flow, corr):
        # if flow.shape[0] == corr.shape[0]
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


# Motion Pattern Aggregation 
class MPA(nn.Module):
    def __init__(self,
                d_model=128):
        super(MPA, self).__init__()
        self.d_model = d_model
        self.layer1 = AttnLayerNoV(d_model, mid_dim=d_model)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, mfs):
        b, c, h, w = mfs[0].shape
        t = len(mfs)

        concat0 = torch.cat(mfs, dim=0).flatten(-2).permute(0, 2, 1)          # [B*T, H*W, C]   [1,2,...,g]
        concat1 = torch.cat(mfs[-1:]*t, dim=0).flatten(-2).permute(0, 2, 1)   # [B*T, H*W, C]   [g,g,g,g,g]

        concat0 = self.layer1(concat0, concat1) 

        #View
        concat0 = concat0.view(-1, h, w, c).permute(0, 3, 1, 2).contiguous()

        return concat0.chunk(t, dim=0)


def single_head_full_attention(q, k, v):
    # q, k, v: [B, L, C]
    assert q.dim() == k.dim() == v.dim() == 3

    scores = torch.matmul(q, k.permute(0, 2, 1)) / (q.size(2) ** .5)  # [B, L, L]
    attn = torch.softmax(scores, dim=2)  # [B, L, L]
    out = torch.matmul(attn, v)  # [B, L, C]

    return out

class AttnLayerNoV(nn.Module):
    def __init__(self, d_model=128, mid_dim=128, no_ffn=False, ffn_dim_expansion=2):
        super(AttnLayerNoV, self).__init__()
        self.dim = d_model
        self.no_ffn = no_ffn

        # multi-head attention
        self.q_proj = nn.Linear(d_model, mid_dim, bias=False)
        self.k_proj = nn.Linear(d_model, mid_dim, bias=False)

        self.merge = nn.Linear(d_model, d_model, bias=False)

        self.norm1 = nn.LayerNorm(d_model)

        # no ffn after self-attn, with ffn after cross-attn
        if not self.no_ffn:
            in_channels = d_model * 2
            self.mlp = nn.Sequential(
                nn.Linear(in_channels, in_channels * ffn_dim_expansion, bias=False),
                nn.GELU(),
                nn.Linear(in_channels * ffn_dim_expansion, d_model, bias=False),
            )

            self.norm2 = nn.LayerNorm(d_model)

    def forward(self, source, target,
                ):
        # source, target: [B, L, C]
        query, key, value = source, target, target

        # single-head attention
        query = self.q_proj(query)  # [B, L, C]
        key = self.k_proj(key)  # [B, L, C]

        message = single_head_full_attention(query, key, value)  # [B, L, C]

        message = self.merge(message)  # [B, L, C]
        message = self.norm1(message)

        if not self.no_ffn:
            message = self.mlp(torch.cat([source, message], dim=-1))
            message = self.norm2(message)

        return source + message

