import torch
import torch.nn as nn
import numpy as np
import numbers
from functools import partial
from timm.models.layers import DropPath, trunc_normal_
from third_party.SparseTransformer.sptr import to_3d_numpy, SparseTrTensor, sparse_self_attention, get_indices_params

class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def cart2sphere(xyz):
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    theta = (torch.atan2(y, x) + np.pi) * 180 / np.pi
    beta = torch.atan2(torch.sqrt(x**2 + y**2), z) * 180 / np.pi
    r = torch.sqrt(x**2 + y**2 + z**2)
    return torch.stack([theta, beta, r], -1)

def exponential_split(xyz, index_0, index_1, relative_position_index, a=0.05*0.25):
    '''
    Mapping functioni from r to idx
    | r         ---> idx    |
    | ...       ---> ...    |
    | [-2a, a)  ---> -2     |
    | [-a, 0)   ---> -1     |
    | [0, a)    ---> 0      |
    | [a, 2a)   ---> 1      |
    | [2a, 4a)  ---> 2      |
    | [4a, 6a)  ---> 3      |
    | [6a, 10a) ---> 4      |
    | [10a, 14a)---> 5      |
    | ...       ---> ...    |
    Starting from 0, the split length will double once used twice.
    '''

    r = xyz[:,2]
    rel_pos = r[index_0.long()] - r[index_1.long()] #[M,3]
    rel_pos_abs = rel_pos.abs()
    flag_float = (rel_pos >= 0).float()
    idx = 2 * torch.floor(torch.log((rel_pos_abs+2*a) / a) / np.log(2)) - 2
    idx = idx + ((3*(2**(idx//2)) - 2)*a <= rel_pos_abs).float()
    idx = idx * (2*flag_float - 1) + (flag_float - 1)
    relative_position_index[:, 2] = idx.long() + 24
    return relative_position_index

# Modified From sptr.VarLengthMultiheadSA (https://github.com/dvlab-research/SparseTransformer)
class SparseMultiheadSASphereConcat(nn.Module):
    def __init__(self, 
        embed_dim, 
        num_heads, 
        indice_key, 
        window_size, 
        window_size_sphere, 
        shift_win=False, 
        pe_type='none', 
        dropout=0., 
        qk_scale=None, 
        qkv_bias=True, 
        algo='native', 
        **kwargs
    ):   
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.indice_key = indice_key
        self.shift_win = shift_win
        self.pe_type = pe_type
        head_dim = embed_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.window_size = to_3d_numpy(window_size)
        self.window_size_sphere = to_3d_numpy(window_size_sphere)

        if pe_type == 'contextual':
            self.rel_query, self.rel_key, self.rel_value = kwargs['rel_query'], kwargs['rel_key'], kwargs['rel_value']

            quant_size = kwargs['quant_size']
            self.quant_size = to_3d_numpy(quant_size)

            quant_size_sphere = kwargs['quant_size_sphere']
            self.quant_size_sphere = to_3d_numpy(quant_size_sphere)

            self.a = kwargs['a']

            quant_grid_length = int((window_size[0] + 1e-4)/ quant_size[0])
            assert int((window_size[0] + 1e-4)/ quant_size[0]) == int((window_size[1] + 1e-4)/ quant_size[1])

            # currently only support rel_query, rel_key and rel_value also equal to True
            assert self.rel_query and self.rel_key and self.rel_value
            num_heads_brc1 = num_heads // 2
            self.num_heads_brc1 = num_heads_brc1

            if self.rel_query:
                self.relative_pos_query_table = nn.Parameter(torch.zeros(2*quant_grid_length-1, 3, num_heads_brc1, head_dim))
                trunc_normal_(self.relative_pos_query_table, std=.02)
            if self.rel_key:
                self.relative_pos_key_table = nn.Parameter(torch.zeros(2*quant_grid_length-1, 3, num_heads_brc1, head_dim))
                trunc_normal_(self.relative_pos_key_table, std=.02)
            if self.rel_value:
                self.relative_pos_value_table = nn.Parameter(torch.zeros(2*quant_grid_length-1, 3, num_heads_brc1, head_dim))
                trunc_normal_(self.relative_pos_value_table, std=.02)
            self.quant_grid_length = quant_grid_length

            quant_grid_length_sphere = int((window_size_sphere[0] + 1e-4) / quant_size_sphere[0])
            assert int((window_size_sphere[0] + 1e-4) / quant_size_sphere[0]) == int((window_size_sphere[1] + 1e-4) / quant_size_sphere[1])
            
            num_heads_brc2 = num_heads - num_heads_brc1
            if self.rel_query:
                self.relative_pos_query_table_sphere = nn.Parameter(torch.zeros(2*quant_grid_length_sphere, 3, num_heads_brc2, head_dim))
                trunc_normal_(self.relative_pos_query_table_sphere, std=.02)
            if self.rel_key:
                self.relative_pos_key_table_sphere = nn.Parameter(torch.zeros(2*quant_grid_length_sphere, 3, num_heads_brc2, head_dim))
                trunc_normal_(self.relative_pos_key_table_sphere, std=.02)
            if self.rel_value:
                self.relative_pos_value_table_sphere = nn.Parameter(torch.zeros(2*quant_grid_length_sphere, 3, num_heads_brc2, head_dim))
                trunc_normal_(self.relative_pos_value_table_sphere, std=.02)
            self.quant_grid_length_sphere = quant_grid_length_sphere

        elif pe_type == 'sine':
            normalize_pos_enc = kwargs.get("normalize_pos_enc", True)
            self.pos_enc = PositionEmbeddingCoordsSine(pos_type="sine",
                                                       d_pos=embed_dim,
                                                       normalize=normalize_pos_enc)
        elif pe_type == "fourier":
            gauss_scale = kwargs.get("gauss_scale", 1.0)
            normalize_pos_enc = kwargs.get("normalize_pos_enc", True)
            self.pos_enc = PositionEmbeddingCoordsSine(pos_type="fourier",
                                                       d_pos=embed_dim,
                                                       gauss_scale=gauss_scale,
                                                       normalize=normalize_pos_enc)

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout, inplace=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout, inplace=True)

    def forward(self, sptr_tensor: SparseTrTensor):
        query, key, value = sptr_tensor.query_feats, sptr_tensor.key_feats, sptr_tensor.value_feats
        assert key is None and value is None
        xyz = sptr_tensor.query_indices[:, 1:]
        batch = sptr_tensor.query_indices[:, 0]

        assert xyz.shape[1] == 3

        N, C = query.shape
        
        qkv = self.qkv(query).reshape(N, 3, self.num_heads, C // self.num_heads).permute(1, 0, 2, 3).contiguous()
        query, key, value = qkv[0], qkv[1], qkv[2] #[N, num_heads, C//num_heads]
        query = query * self.scale

        xyz_sphere = cart2sphere(xyz)
        index_params = sptr_tensor.find_indice_params(self.indice_key)
        if index_params is None:
            index_0, index_0_offsets, n_max, index_1, index_1_offsets, sort_idx = get_indices_params(
                xyz, 
                batch, 
                self.window_size, 
                self.shift_win
            )
            index_0_sphere, index_0_offsets_sphere, n_max_sphere, index_1_sphere, index_1_offsets_sphere, sort_idx_sphere = get_indices_params(
                xyz_sphere, 
                batch, 
                self.window_size_sphere, 
                self.shift_win
            )
            sptr_tensor.indice_dict[self.indice_key] = (
                index_0, 
                index_0_offsets, 
                n_max, 
                index_1, 
                index_1_offsets, 
                sort_idx, 
                self.window_size, 
                self.shift_win, 
                index_0_sphere, 
                index_0_offsets_sphere, 
                n_max_sphere, 
                index_1_sphere, 
                index_1_offsets_sphere, 
                sort_idx_sphere
            )
        else:
            index_0, index_0_offsets, n_max, index_1, index_1_offsets, sort_idx, window_size, shift_win, \
                index_0_sphere, index_0_offsets_sphere, n_max_sphere, index_1_sphere, index_1_offsets_sphere, sort_idx_sphere = index_params
            assert (window_size == self.window_size) and (shift_win == self.shift_win), "window_size and shift_win must be the same for sptr_tensors with the same indice_key: {}".format(self.indice_key)
            assert (window_size_sphere == self.window_size_sphere), "window_size and shift_win must be the same for sptr_tensors with the same indice_key: {}".format(self.indice_key)

        kwargs = {"query": query[:, :self.num_heads_brc1].contiguous().float(),
            "key": key[:, :self.num_heads_brc1].contiguous().float(), 
            "value": value[:, :self.num_heads_brc1].contiguous().float(),
            "xyz": xyz.float(),
            "index_0": index_0.int(),
            "index_0_offsets": index_0_offsets.int(),
            "n_max": n_max,
            "index_1": index_1.int(), 
            "index_1_offsets": index_1_offsets.int(),
            "sort_idx": sort_idx,
            "window_size": self.window_size,
            "shift_win": self.shift_win,
            "pe_type": self.pe_type,
        }
        if self.pe_type == 'contextual':
            kwargs.update({
                "rel_query": self.rel_query,
                "rel_key": self.rel_key,
                "rel_value": self.rel_value,
                "quant_size": self.quant_size,
                "quant_grid_length": self.quant_grid_length,
                "relative_pos_query_table": self.relative_pos_query_table.float(),
                "relative_pos_key_table": self.relative_pos_key_table.float(),
                "relative_pos_value_table": self.relative_pos_value_table.float()
            })
        out1 = sparse_self_attention(**kwargs)

        kwargs = {"query": query[:, self.num_heads_brc1:].contiguous().float(),
            "key": key[:, self.num_heads_brc1:].contiguous().float(), 
            "value": value[:, self.num_heads_brc1:].contiguous().float(),
            "xyz": xyz_sphere.float(),
            "index_0": index_0_sphere.int(),
            "index_0_offsets": index_0_offsets_sphere.int(),
            "n_max": n_max_sphere,
            "index_1": index_1_sphere.int(), 
            "index_1_offsets": index_1_offsets_sphere.int(),
            "sort_idx": sort_idx_sphere,
            "window_size": self.window_size_sphere,
            "shift_win": self.shift_win,
            "pe_type": self.pe_type,
        }
        if self.pe_type == 'contextual':
            kwargs.update({
                "rel_query": self.rel_query,
                "rel_key": self.rel_key,
                "rel_value": self.rel_value,
                "quant_size": self.quant_size_sphere,
                "quant_grid_length": self.quant_grid_length_sphere,
                "relative_pos_query_table": self.relative_pos_query_table_sphere.float(),
                "relative_pos_key_table": self.relative_pos_key_table_sphere.float(),
                "relative_pos_value_table": self.relative_pos_value_table_sphere.float(),
                "split_func": partial(exponential_split, a=self.a),
            })
        out2 = sparse_self_attention(**kwargs)

        x = torch.cat([out1, out2], 1).view(N, C)

        x = self.proj(x)
        x = self.proj_drop(x) #[N, C]

        output_tensor = SparseTrTensor(x, sptr_tensor.query_indices, sptr_tensor.spatial_shape, sptr_tensor.batch_size)

        return output_tensor


class SphereFormer(nn.Module):
    def __init__(self, 
        dim, 
        num_heads, 
        window_size, 
        window_size_sphere, 
        quant_size, 
        quant_size_sphere, 
        indice_key,
        pe_type='contextual',
        rel_query=True, 
        rel_key=False, 
        rel_value=False,
        drop_path=0.0,
        mlp_ratio=4.0, 
        qkv_bias=True, 
        qk_scale=None, 
        act_layer=nn.GELU, 
        norm_layer=nn.LayerNorm, 
        a=0.05*0.25,
    ):
        super().__init__()
        self.window_size = window_size

        self.norm1 = norm_layer(dim)
        self.attn = SparseMultiheadSASphereConcat(
            dim, 
            num_heads=num_heads, 
            indice_key=indice_key, 
            window_size=window_size, 
            window_size_sphere=window_size_sphere, 
            pe_type=pe_type,
            quant_size=quant_size, 
            quant_size_sphere=quant_size_sphere, 
            rel_query=rel_query, 
            rel_key=rel_key, 
            rel_value=rel_value, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale,
            a=a,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, feats, xyz, batch):
        # feats: [N, c]
        # xyz: [N, 3]
        # batch: [N]

        short_cut = feats
        feats = self.norm1(feats)

        sptr_tensor = SparseTrTensor(feats, torch.cat([batch[:, None], xyz], -1), spatial_shape=None, batch_size=None)
        sptr_tensor = self.attn(sptr_tensor)
        feats = sptr_tensor.query_feats

        feats = short_cut + self.drop_path(feats)
        feats = feats + self.drop_path(self.mlp(self.norm2(feats)))

        return feats
