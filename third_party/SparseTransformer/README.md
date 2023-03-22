# SpTr: PyTorch Spatially Sparse Transformer Library

This library provides a **fast** and **memory-efficient** implementation for sparse transformer with **varying token numbers** (e.g., window transformer for 3D point cloud).

This library has been used by the following works:

* Spherical Transformer for LiDAR-based 3D Recognition (CVPR 2023): \[Paper\] [\[Code\]](https://github.com/dvlab-research/SphereFormer)

* Stratified Transformer for 3D Point Cloud Segmentation (CVPR 2022): [\[Paper\]](https://openaccess.thecvf.com/content/CVPR2022/papers/Lai_Stratified_Transformer_for_3D_Point_Cloud_Segmentation_CVPR_2022_paper.pdf) [\[Code\]](https://github.com/dvlab-research/Stratified-Transformer)

## Installation
### Install Dependency
```
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch_scatter
pip install torch_geometric
```

### Compile sptr
```
python3 setup.py install
```


## Usage
SpTr can be easily used in most current transformer-based 3D point cloud networks, with only several minor modifications. First, define the attention module `sptr.VarLengthMultiheadSA`. Then, wrap the input features and indices into `sptr.SparseTrTensor`, and foward it into the module. That's all. A simple example is as follows. For more complex usage, you can refer to the code of above works (e.g., SphereFormer, StratifiedFormer).
### Example
```
import sptr

# Define module
dim = 48
num_heads = 3
indice_key = 'sptr_0'
window_size = np.array([0.4, 0.4, 0.4])  # can also be integers for voxel-based methods
shift_win = False  # whether to adopt shifted window
self.attn = sptr.VarLengthMultiheadSA(
    dim, 
    num_heads, 
    indice_key, 
    window_size, 
    shift_win
)

# Wrap the input features and indices into SparseTrTensor. Note: indices can be either intergers for voxel-based methods or floats (i.e., xyz) for point-based methods
# feats: [N, C], indices: [N, 4] with batch indices in the 0-th column
input_tensor = sptr.SparseTrTensor(feats, indices, spatial_shape=None, batch_size=None)
output_tensor = self.attn(input_tensor)

# Extract features from output tensor
output_feats = output_tensor.query_feats
```

## Authors

Xin Lai (a Ph.D student at CSE CUHK) - Initial CUDA implementation, maintainance.

Fanbin Lu (a Ph.D student at CSE CUHK) - Improve CUDA implementation, maintainance.

Yukang Chen (a Ph.D student at CSE CUHK) - Maintainance. 


## Cite

If you find this project useful, please consider citing
```
@inproceedings{lai2023spherical,
  title={Spherical Transformer for LiDAR-based 3D Recognition},
  author={Lai, Xin and Chen, Yukang and Lu, Fanbin and Liu, Jianhui and Jia, Jiaya},
  booktitle={CVPR},
  year={2023}
}
```
```
@inproceedings{lai2022stratified,
  title={Stratified transformer for 3d point cloud segmentation},
  author={Lai, Xin and Liu, Jianhui and Jiang, Li and Wang, Liwei and Zhao, Hengshuang and Liu, Shu and Qi, Xiaojuan and Jia, Jiaya},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={8500--8509},
  year={2022}
}
```

## License

This project is licensed under the Apache license 2.0 License.
