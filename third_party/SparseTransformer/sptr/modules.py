import torch
import torch.nn as nn
import numpy as np
import numbers
from timm.models.layers import DropPath, trunc_normal_
from . import SparseTrTensor
from .functional import attention_step1, attention_step2, dot_prod_with_idx, dot_prod_with_idx_all, attention_step2_with_rel_pos_value
from .utils import to_3d_numpy, scatter_softmax_csr, get_indices_params
from .position_embedding import PositionEmbeddingCoordsSine

def sparse_self_attention(query, 
    key, 
    value, 
    xyz,
    index_0,
    index_0_offsets,
    n_max,
    index_1,
    index_1_offsets,
    sort_idx,
    window_size, 
    shift_win, 
    pe_type='none', 
    rel_query=False, 
    rel_key=False, 
    rel_value=False, 
    quant_size=None, 
    quant_grid_length=None, 
    relative_pos_query_table=None, 
    relative_pos_key_table=None, 
    relative_pos_value_table=None,
    split_func=None,
):
    
    query = query[sort_idx]
    key = key[sort_idx]
    value = value[sort_idx]
    xyz_ctg = xyz[sort_idx]
    
    if pe_type == 'contextual' and rel_query and rel_key:
        window_size = torch.from_numpy(window_size).float().cuda()
        shift_size = 1/2 * window_size if shift_win else 0.0
        xyz_quant = (xyz_ctg - xyz_ctg.min(0)[0] + shift_size) % window_size
        xyz_quant = torch.div(xyz_quant, torch.from_numpy(quant_size).float().cuda(), rounding_mode='floor') #[N, 3]
        relative_position = xyz_quant[index_0.long()] - xyz_quant[index_1.long()] #[M, 3]
        relative_position_index = relative_position + quant_grid_length - 1
        if split_func:
            relative_position_index = split_func(xyz_ctg, index_0, index_1, relative_position_index.clone())
            relative_position_index = torch.clamp(relative_position_index, 0, 2*quant_grid_length-1)
        
        relative_position_index = relative_position_index.int()
        attn_flat = dot_prod_with_idx_all(query, index_0, index_0_offsets, key, index_1, index_1_offsets, relative_pos_query_table, relative_pos_key_table, relative_position_index, n_max)
    else:
        attn_flat = attention_step1(query, key, index_0, index_0_offsets, index_1, index_1_offsets, n_max)

    softmax_attn_flat = scatter_softmax_csr(src=attn_flat, indptr=index_0_offsets.long(), dim=0) #[M, num_heads]

    if pe_type == 'contextual' and rel_value:
        x = attention_step2_with_rel_pos_value(softmax_attn_flat, value, index_0, index_0_offsets, n_max, index_1, index_1_offsets, relative_pos_value_table, relative_position_index)
    else:
        x = attention_step2(softmax_attn_flat, value, index_0, index_0_offsets, index_1, index_1_offsets, n_max)
    
    out = torch.empty_like(x)
    out[sort_idx] = x
    x = out
    return x


class VarLengthMultiheadSA(nn.Module):
    def __init__(self, embed_dim, num_heads, indice_key, window_size, shift_win=False, pe_type='none', dropout=0., qk_scale=None, qkv_bias=True, algo='native', **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.indice_key = indice_key
        self.shift_win = shift_win
        self.pe_type = pe_type
        head_dim = embed_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.window_size = to_3d_numpy(window_size)

        if pe_type == 'contextual':
            self.rel_query, self.rel_key, self.rel_value = kwargs['rel_query'], kwargs['rel_key'], kwargs['rel_value']

            quant_size = kwargs['quant_size']
            self.quant_size = to_3d_numpy(quant_size)

            quant_grid_length = int((window_size[0] + 1e-4)/ quant_size[0])
            assert int((window_size[0] + 1e-4)/ quant_size[0]) == int((window_size[1] + 1e-4)/ quant_size[1])
            # currently only support rel_query, rel_key and rel_value also equal to True
            assert self.rel_query and self.rel_key and self.rel_value
            if self.rel_query:
                self.relative_pos_query_table = nn.Parameter(torch.zeros(2*quant_grid_length-1, 3, num_heads, head_dim))
                trunc_normal_(self.relative_pos_query_table, std=.02)
            if self.rel_key:
                self.relative_pos_key_table = nn.Parameter(torch.zeros(2*quant_grid_length-1, 3, num_heads, head_dim))
                trunc_normal_(self.relative_pos_key_table, std=.02)
            if self.rel_value:
                self.relative_pos_value_table = nn.Parameter(torch.zeros(2*quant_grid_length-1, 3, num_heads, head_dim))
                trunc_normal_(self.relative_pos_value_table, std=.02)
            self.quant_grid_length = quant_grid_length
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

        self.q = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.k = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.v = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout, inplace=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout, inplace=True)

    def forward(self, sptr_tensor: SparseTrTensor):
        query, key, value = sptr_tensor.query_feats, sptr_tensor.key_feats, sptr_tensor.value_feats
        if key is None:
            key = query.clone()
        if value is None:
            value = query.clone()
        xyz = sptr_tensor.query_indices[:, 1:]
        batch = sptr_tensor.query_indices[:, 0]

        if self.pe_type in ['sine', 'fourier']:
            offset = batch.unique_consecutive(return_counts=True, dim=0)[1].cumsum(-1)
            assert offset.shape[0] == batch.max().item() + 1, "batch must be sorted and consecutive"
            offset = torch.cat([offset.new_zeros(1), offset], 0)
            pos_emb = []
            for i in range(offset.shape[0] - 1):
                s, e = offset[i], offset[i+1]
                xyz_i = xyz[s:e]
                min_i, max_i = xyz_i.min(0)[0], xyz_i.max(0)[0]
                pos_emb_i = self.pos_enc(xyz_i[None, ...], 
                                    input_range=[min_i[None, ...], max_i[None, ...]])
                pos_emb.append(pos_emb_i[0].permute(1,0).contiguous())
                
            pos_emb = torch.cat(pos_emb, 0)

            query += pos_emb
            key += pos_emb

        assert xyz.shape[1] == 3

        N, C = query.shape
        
        query = self.q(query).reshape(N, self.num_heads, C // self.num_heads)
        key = self.k(key).reshape(N, self.num_heads, C // self.num_heads)
        value = self.v(value).reshape(N, self.num_heads, C // self.num_heads)
        query = query * self.scale
        
        index_params = sptr_tensor.find_indice_params(self.indice_key)
        if index_params is None:
            index_0, index_0_offsets, n_max, index_1, index_1_offsets, sort_idx = get_indices_params(xyz, batch, self.window_size, self.shift_win)
            sptr_tensor.indice_dict[self.indice_key] = (index_0, index_0_offsets, n_max, index_1, index_1_offsets, sort_idx, self.window_size, self.shift_win)
        else:
            index_0, index_0_offsets, n_max, index_1, index_1_offsets, sort_idx, window_size, shift_win = index_params
            assert (window_size == self.window_size) and (shift_win == self.shift_win), "window_size and shift_win must be the same for sptr_tensors with the same indice_key: {}".format(self.indice_key)

        kwargs = {"query": query.float(),
            "key": key.float(), 
            "value": value.float(),
            "xyz": xyz.float(),
            "index_0": index_0.int(),
            "index_0_offsets": index_0_offsets.int(),
            "n_max": n_max,
            "index_1": index_1.int(), 
            "index_1_offsets": index_1_offsets.int(),
            "sort_idx": sort_idx,
            "window_size": self.window_size,
            "shift_win": self.shift_win
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

        x = sparse_self_attention(**kwargs)
        x = x.view(N, C)

        x = self.proj(x)
        x = self.proj_drop(x) #[N, C]

        output_tensor = SparseTrTensor(x, sptr_tensor.query_indices, sptr_tensor.spatial_shape, sptr_tensor.batch_size)

        return output_tensor

