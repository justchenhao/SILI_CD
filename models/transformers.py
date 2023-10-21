import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


class BIT(nn.Module):
    def __init__(self, dim=32, token_len=4, with_pos='learned',
                 tokenizer_mode='filter', pool_size=2, pool_mode='ave',
                 token_trans=True, encoder_depth=1, decoder_depth=1,
                 recurrent_num=1):
        super(BIT, self).__init__()

        self.token_trans = token_trans
        self.tokenizer_mode = tokenizer_mode
        if self.tokenizer_mode == 'pool':
            #  使用pool的方式，把特征图降采样到每个尺度
            self.pooling_size = pool_size
            self.pool_mode = pool_mode
            self.token_len = self.pooling_size * self.pooling_size
        elif self.tokenizer_mode == 'filter':
            self.conv_a = nn.Conv2d(dim, token_len, kernel_size=1,
                                    padding=0, bias=False)
            self.token_len = token_len
        else:
            raise NotImplementedError(tokenizer_mode)

        self.with_pos = with_pos
        self.recurrent_num = recurrent_num
        if with_pos is 'learned':
            self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len * 2, dim))

        self.dim = dim
        mlp_dim = 2*dim
        self.transformer = Transformer(dim=self.dim, depth=encoder_depth, heads=8, dim_head=64, mlp_dim=mlp_dim, dropout=0)
        self.transformer_decoder = TransformerDecoder(dim=self.dim, depth=decoder_depth, heads=8, dim_head=64, mlp_dim=mlp_dim, dropout=0)

    def _forward_semantic_tokens(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)
        return tokens

    def _forward_transformer(self, x):
        if self.with_pos:
            x += self.pos_embedding
        x = self.transformer(x)
        return x

    def _forward_transformer_decoder(self, x, m):
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

    def _forward_reshape_tokens(self, x):
        # b,c,h,w = x.shape
        if self.pool_mode is 'max':
            x = F.adaptive_max_pool2d(x, [self.pooling_size, self.pooling_size])
        elif self.pool_mode is 'ave':
            x = F.adaptive_avg_pool2d(x, [self.pooling_size, self.pooling_size])
        tokens = rearrange(x, 'b c h w -> b (h w) c')
        return tokens

    def forward(self, x1, x2):
        for i in range(self.recurrent_num):
            if self.tokenizer_mode == 'filter':
                token1 = self._forward_semantic_tokens(x1)
                token2 = self._forward_semantic_tokens(x2)
            elif self.tokenizer_mode == 'pool':
                token1 = self._forward_reshape_tokens(x1)
                token2 = self._forward_reshape_tokens(x2)
            else:
                raise NotImplementedError(self.tokenizer_mode)

            if self.token_trans:
                tokens = torch.cat([token1, token2], dim=1)
                tokens = self._forward_transformer(tokens)
                token1, token2 = tokens.chunk(2, dim=1)

            x1 = self._forward_transformer_decoder(x1, token1)
            x2 = self._forward_transformer_decoder(x2, token2)

        return x1, x2


class AttentionInterface(nn.Module):
    def __init__(self, dim=64, heads=1, ds=2):
        super(AttentionInterface, self).__init__()
        self.att = Attention(dim=dim, heads=heads, dim_head=64, dropout=0.)
        self.ds = ds
        if ds != 1:
            self.downsample = nn.AvgPool2d(kernel_size=ds, stride=ds)
    def forward(self, x1, x2):
        # b, c, h, w
        B, C, H, W = x1.shape
        x = torch.concat([x1, x2], dim=-1)
        shortcut = x
        if self.ds != 1:
            x = self.downsample(x)
        x = rearrange(x, 'b c h w -> b (h w) c')

        x = self.att(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H//self.ds)
        if self.ds != 1:
            x = F.interpolate(x, size=(H, 2*W), mode='bilinear', align_corners=True)
        x = x + shortcut
        x1, x2 = torch.split(x, [W, W], dim=-1)

        return x1, x2


##########################################################################################
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout, softmax=True):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual2(PreNorm2(dim, Cross_Attention(dim, heads = heads,
                                                        dim_head = dim_head, dropout = dropout,
                                                        softmax=softmax))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, m, mask = None):
        """target(query), memory"""
        for attn, ff in self.layers:
            x = attn(x, m, mask = mask)
            x = ff(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Cross_Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., softmax=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.softmax = softmax
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, m, mask = None):

        b, n, _, h = *x.shape, self.heads
        q = self.to_q(x)
        k = self.to_k(m)
        v = self.to_v(m)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), [q,k,v])

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        if self.softmax:
            attn = dots.softmax(dim=-1)
        else:
            attn = dots
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class Residual2(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, x2, **kwargs):
        return self.fn(x, x2, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNorm2(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, x2, **kwargs):
        return self.fn(self.norm(x), self.norm(x2), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
