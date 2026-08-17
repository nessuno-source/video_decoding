import torch
import torch.nn as nn

VGG_CHANNELS = [64, 128, 256, 512, 512]     # the five ReLU layers used by the early stream
GRID = 7                                     # spatial grid kept per layer
N_FRAMES = 6


class FactorizedAttnPool(nn.Module):

    def __init__(self, C=768, num_queries=1, num_heads=4, T=8, H=14, W=14, learnable_pos=True):
        super().__init__()
        assert C % num_heads == 0
        self.nq, self.nh, self.C, self.T, self.HW = num_queries, num_heads, C, T, H * W
        self.pos_t = nn.Parameter(torch.randn(T, C) * C ** -0.5) if learnable_pos else None
        self.pos_s = nn.Parameter(torch.randn(H * W, C) * C ** -0.5) if learnable_pos else None
        self.q = nn.Parameter(torch.randn(num_queries, C) * C ** -0.5)
        self.k = nn.Linear(C, C, bias=False)
        self.v = nn.Linear(C, C, bias=False)
        self.scale = (C // num_heads) ** -0.5

    def forward(self, tokens, return_attn=False):          # (B, T*HW, C)
        B, N, C = tokens.shape
        kin = tokens
        if self.pos_t is not None:
            pos = (self.pos_t[:, None, :] + self.pos_s[None, :, :]).reshape(1, N, C)
            kin = tokens + pos
        K, V = self.k(kin), self.v(tokens)                 # positions only in the keys
        split = lambda z: z.view(B, N, self.nh, C // self.nh).transpose(1, 2)
        Kh, Vh = split(K), split(V)
        Qh = self.q.view(self.nq, self.nh, C // self.nh).permute(1, 0, 2)[None]
        attn = (Qh @ Kh.transpose(-2, -1) * self.scale).softmax(-1)
        out = (attn @ Vh).transpose(1, 2).reshape(B, self.nq * C)
        return (out, attn.mean(1)) if return_attn else out


class TemporalAttnPool(nn.Module):

    def __init__(self, ch=VGG_CHANNELS, T=N_FRAMES, S=GRID * GRID, per_cell=True):
        super().__init__()
        self.ch, self.T, self.S = list(ch), T, S
        self.sizes = [c * S for c in self.ch]
        self.q = nn.ParameterList([nn.Parameter(torch.randn(S if per_cell else 1, c) * c ** -0.5)
                                   for c in self.ch])
        self.k = nn.ModuleList([nn.Linear(c, c, bias=False) for c in self.ch])
        self.pos_t = nn.ParameterList([nn.Parameter(torch.randn(T, c) * c ** -0.5)
                                       for c in self.ch])
        self.scale = [c ** -0.5 for c in self.ch]

    def forward(self, x):                                   # (B, T, sum(C_l * 49))
        B, T, D = x.shape
        assert D == sum(self.sizes), (D, sum(self.sizes))
        out, off = [], 0
        for li, (c, sz) in enumerate(zip(self.ch, self.sizes)):
            blk = x[:, :, off:off + sz].reshape(B, T, c, self.S)     # (B,T,C,S)
            tok = blk.permute(0, 3, 1, 2)                            # (B,S,T,C)
            K = self.k[li](tok + self.pos_t[li][None, None])         # positions only in the keys
            q = self.q[li]
            # each cell compares ITS keys with ITS query: no spatial mixing
            w = (torch.einsum("bstc,sc->bst", K, q) if q.shape[0] == self.S
                 else torch.einsum("bstc,c->bst", K, q[0]))
            w = (w * self.scale[li]).softmax(-1)                     # (B,S,T)
            agg = (tok * w.unsqueeze(-1)).sum(2)                     # (B,S,C)
            out.append(agg.transpose(1, 2).reshape(B, c * self.S))
            off += sz
        return torch.cat(out, 1)
