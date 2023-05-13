
# Spherical Transformer for LiDAR-based 3D Recognition (CVPR 2023)

This is the official PyTorch implementation of *object detection* for **SphereFormer** (CVPR 2023).

**Spherical Transformer for LiDAR-based 3D Recognition** [\[Paper\]](https://arxiv.org/pdf/2303.12766.pdf)

Xin Lai, Yukang Chen, Fanbin Lu, Jianhui Liu, Jiaya Jia 

# Get Started

Clone the repo via
```
git clone https://github.com/dvlab-research/SphereFormer.git --recursive && cd SphereFormer/detection/
```

This implementataion is built on OpenPCDet (https://github.com/open-mmlab/OpenPCDet). Please strictly follow its official guidance for installation and data preparation.

In addition to installing `OpenPCDet`, follow the following command to install `Sparse Transformer (SpTr)`.
```
cd tools/third_party/SparseTransformer && python setup.py install
```

# Training
```
bash scripts/dist_train.sh 4 --cfg_file cfgs/nuscenes_models/cbgs_voxel0075_res3d_centerpoint_sphereformer.yaml
```