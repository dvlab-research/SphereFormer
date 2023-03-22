import numbers
import torch
import numpy as np
from torch_scatter import segment_csr, gather_csr
from torch_geometric.nn import voxel_grid
from . import precompute_all


def to_3d_numpy(size):
    if isinstance(size, numbers.Number):
        size = np.array([size, size, size]).astype(np.float32)
    elif isinstance(size, list):
        size = np.array(size)
    elif isinstance(size, np.ndarray):
        size = size
    else:
        raise ValueError("size is either a number, or a list, or a np.ndarray")
    return size

def grid_sample(pos, batch, size, start, return_p2v=True, return_counts=True, return_unique=False):
    # pos: float [N, 3]
    # batch: long [N]
    # size: float [3, ]
    # start: float [3, ] / None

    cluster = voxel_grid(pos, batch, size, start=start) #[N, ]

    if return_p2v == False and return_counts == False:
        unique, cluster = torch.unique(cluster, sorted=True, return_inverse=True)
        return cluster

    unique, cluster, counts = torch.unique(cluster, sorted=True, return_inverse=True, return_counts=True)

    if return_p2v == False and return_counts == True:
        return cluster, counts.max().item(), counts

    # obtain p2v_map
    n = unique.shape[0]
    k = counts.max().item()
    p2v_map = cluster.new_zeros(n, k) #[n, k]
    mask = torch.arange(k).cuda().unsqueeze(0) < counts.unsqueeze(-1) #[n, k]
    p2v_map[mask] = torch.argsort(cluster)

    if return_unique:
        return cluster, p2v_map, counts, unique

    return cluster, p2v_map, counts

def get_indices_params(xyz, batch, window_size, shift_win: bool):
    
    if isinstance(window_size, list) or isinstance(window_size, np.ndarray):
        window_size = torch.from_numpy(window_size).type_as(xyz).to(xyz.device)
    else:
        window_size = torch.tensor([window_size]*3).type_as(xyz).to(xyz.device)
    
    if shift_win:
        v2p_map, k, counts = grid_sample(xyz+1/2*window_size, batch, window_size, start=xyz.min(0)[0], return_p2v=False, return_counts=True)
    else:
        v2p_map, k, counts = grid_sample(xyz, batch, window_size, start=None, return_p2v=False, return_counts=True)

    v2p_map, sort_idx = v2p_map.sort()

    n = counts.shape[0]
    N = v2p_map.shape[0]

    n_max = k
    index_0_offsets, index_1_offsets, index_0, index_1 = precompute_all(N, n, n_max, counts)
    index_0 = index_0.long()
    index_1 = index_1.long()

    return index_0, index_0_offsets, n_max, index_1, index_1_offsets, sort_idx

def scatter_softmax_csr(src: torch.Tensor, indptr: torch.Tensor, dim: int = -1):
    ''' src: (N, C),
        index: (Ni+1, ), [0, n0^2, n0^2+n1^2, ...]
    '''
    max_value_per_index = segment_csr(src, indptr, reduce='max')
    max_per_src_element = gather_csr(max_value_per_index, indptr)
    
    recentered_scores = src - max_per_src_element
    recentered_scores_exp = recentered_scores.exp_()

    sum_per_index = segment_csr(
        recentered_scores_exp, indptr, reduce='sum')
    
    normalizing_constants = gather_csr(sum_per_index, indptr)

    return recentered_scores_exp.div(normalizing_constants)
