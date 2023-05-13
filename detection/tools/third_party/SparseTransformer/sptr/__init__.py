from .functional import *
from .utils import *

class SparseTrTensor(object):
    def __init__(self, query_feats, query_indices, spatial_shape, batch_size, key_feats=None, value_feats=None, key_indices=None):
        """
        Args:
            query_feats: [num_points, num_features] feature tensor
            indices: [num_points, ndim + 1] indice tensor. batch index saved in indices[:, 0]
            spatial_shape: spatial shape of your sparse data
            batch_size: batch size of your sparse data
        """
        self.query_feats = query_feats
        self.key_feats = key_feats
        self.value_feats = value_feats
        self.query_indices = query_indices
        self.key_indices = key_indices
        self.spatial_shape = spatial_shape
        self.batch_size = batch_size
        self.indice_dict = {}

    @property
    def spatial_size(self):
        return np.prod(self.spatial_shape)

    def find_indice_params(self, key):
        if key is None:
            return None
        if key in self.indice_dict:
            return self.indice_dict[key]
        return None

from .modules import VarLengthMultiheadSA, sparse_self_attention
