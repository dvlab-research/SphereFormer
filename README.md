[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spherical-transformer-for-lidar-based-3d/3d-semantic-segmentation-on-semantickitti)](https://paperswithcode.com/sota/3d-semantic-segmentation-on-semantickitti?p=spherical-transformer-for-lidar-based-3d)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spherical-transformer-for-lidar-based-3d/lidar-semantic-segmentation-on-nuscenes)](https://paperswithcode.com/sota/lidar-semantic-segmentation-on-nuscenes?p=spherical-transformer-for-lidar-based-3d)

# Spherical Transformer for LiDAR-based 3D Recognition (CVPR 2023)

This is the official PyTorch implementation of **SphereFormer** (CVPR 2023).

**Spherical Transformer for LiDAR-based 3D Recognition** [\[Paper\]](https://arxiv.org/pdf/2303.12766.pdf)

Xin Lai, Yukang Chen, Fanbin Lu, Jianhui Liu, Jiaya Jia 

# Highlight 
1. **SphereFormer** is a plug-and-play transformer module. We develop **radial window attention**, which significantly boosts the segmentation performance of **distant points**, e.g., from 13.3% to 30.4% mIoU on nuScenes lidarseg *val* set. 
2. It achieves superior performance on various **outdoor semantic segmentation benchmarks**, e.g., nuScenes, SemanticKITTI, Waymo, and also shows competitive results on **nuScenes detection** dataset.
3. This repository employs a **fast** and **memory-efficient** library for sparse transformer with **varying token numbers**, [**SparseTransformer**](https://github.com/dvlab-research/SparseTransformer).

<div align="center">
  <img src="figs/figure.jpg"/>
</div>

# Get Started

For *object deteciton*, please go to the `detection/` directory. (or click [Here](detection/README.md))

The below guide is for *semantic segmentation*.

## Environment

Install dependencies (we test on python=3.7.9, pytorch==1.8.0, cuda==11.1, gcc==7.5.0)
```
git clone https://github.com/dvlab-research/SphereFormer.git --recursive
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch_scatter==2.0.9
pip install torch_geometric==1.7.2
pip install spconv-cu114==2.1.21
pip install torch_sparse==0.6.12 cumm-cu114==0.2.8 torch_cluster==1.5.9
pip install tensorboard timm termcolor tensorboardX
```

Install `sptr`
```
cd third_party/SparseTransformer && python setup.py install
```

Note: Make sure you have installed `gcc` and `cuda`, and `nvcc` can work (if you install cuda by conda, it won't provide nvcc and you should install cuda manually.)

## Datasets Preparation

### nuScenes
Download the nuScenes dataset from [here](https://www.nuscenes.org/nuscenes#download). Unzip and arrange it as follows. Then fill in the `data_root` entry in the .yaml configuration file.
```
nuscenes/
|--- v1.0-trainval/
|--- samples/
|------- LIDAR_TOP/
|--- lidarseg/
|------- v1.0-trainval/
```
Then, fill in the `data_path` and `save_dir` in `data/nuscenes_preprocess_infos.py`, then generate the infos by
```
pip install nuscenes-devkit pyquaternion
cd data && python nuscenes_preprocess_infos.py
```

### SemanticKITTI
Download the SemanticKIITI dataset from [here](http://www.semantic-kitti.org/dataset.html#download). Unzip and arrange it as follows. Then fill in the `data_root` entry in the .yaml configuration file.
```
dataset/
|--- sequences/
|------- 00/
|------- 01/
|------- 02/
|------- 03/
|------- .../
```

### Waymo Open Dataset
Download the Waymo Open Dataset from [here](https://waymo.com/open/). Unzip and arrange it as follows. Then fill in the `data_root` entry in the .yaml configuration file.
```
waymo/
|--- training/
|--- validation/
|--- testing/
```
Then, transfer the raw files into the format of SemanticKITTI as follows. (Note: do not use GPU here, and CPU works well already)
```
cd data/waymo_to_semanticKITTI
CUDA_VISIBLE_DEVICES="" python convert.py --load_dir [YOUR_DATA_ROOT] --save_dir [YOUR_SAVE_ROOT]
```

## Training

### nuScenes
```
python train.py --config config/nuscenes/nuscenes_unet32_spherical_transformer.yaml
```

### SemanticKITTI
```
python train.py --config config/semantic_kitti/semantic_kitti_unet32_spherical_transformer.yaml
```

### Waymo Open Dataset
```
python train.py --config config/waymo/waymo_unet32_spherical_transformer.yaml
```

## Validation
For validation, you need to modify the `.yaml` config file. (1) fill in the `weight` with the path of model weight (`.pth` file); (2) set `val` to `True`; (3) for testing-time augmentation, set `use_tta` to `True` and set `vote_num` accordingly. After that, run the following command. 
```
python train.py --config [YOUR_CONFIG_PATH]
```

## Pre-trained Models


| dataset | Val mIoU (tta) | Val mIoU | mIoU_close | mIoU_medium | mIoU_distant |  Download  |
|---------------|:----:|:----:|:----:|:----:|:----:|:-----------:|
| [nuScenes](config/nuscenes/nuscenes_unet32_spherical_transformer.yaml) | 79.5 | 78.4 | 80.8 | 60.8 | 30.4 | [Model Weight](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155154502_link_cuhk_edu_hk/Ebj08nZvE5lPpRn1ALgkcKwBjEQ5lrQFhx-yR2cbi9Cy-A?e=D3N3ge) |
| [SemanticKITTI](config/semantic_kitti/semantic_kitti_unet32_spherical_transformer.yaml) | 69.0 | 67.8 | 68.6 | 60.4 | 17.8 | [Model Weight](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155154502_link_cuhk_edu_hk/EXsr5RdFzd9Lj7_T8L0dCagBZCDmbe5DtcZ8ipf1CfC58w?e=KxGpLV) |
| [Waymo Open Dataset](config/waymo/waymo_unet32_spherical_transformer.yaml) | 70.8 | 69.9 | 70.3 | 68.6 | 61.9 | N/A |

Note: Pre-trained weights on Waymo Open Dataset are not released due to the regulations. 

# SpTr Library
The `SpTr` library is highly recommended for sparse transformer, particularly for 3D point cloud attention. It is **fast**, **memory-efficient** and **easy-to-use**. The github repository is https://github.com/dvlab-research/SparseTransformer.git.

# Citation
If you find this project useful, please consider citing:

```
@inproceedings{lai2023spherical,
  title={Spherical Transformer for LiDAR-based 3D Recognition},
  author={Lai, Xin and Chen, Yukang and Lu, Fanbin and Liu, Jianhui and Jia, Jiaya},
  booktitle={CVPR},
  year={2023}
}
```

# Our Works on 3D Point Cloud

* **Spherical Transformer for LiDAR-based 3D Recognition (CVPR 2023)** [\[Paper\]](https://arxiv.org/pdf/2303.12766.pdf) [\[Code\]](https://github.com/dvlab-research/SphereFormer) : A plug-and-play transformer module that boosts performance for distant region (for 3D LiDAR point cloud)

* **Stratified Transformer for 3D Point Cloud Segmentation (CVPR 2022)**: [\[Paper\]](https://openaccess.thecvf.com/content/CVPR2022/papers/Lai_Stratified_Transformer_for_3D_Point_Cloud_Segmentation_CVPR_2022_paper.pdf) [\[Code\]](https://github.com/dvlab-research/Stratified-Transformer) : Point-based window transformer for 3D point cloud segmentation

* **SparseTransformer (SpTr) Library** [\[Code\]](https://github.com/dvlab-research/SparseTransformer) : A fast, memory-efficient, and easy-to-use library for sparse transformer with varying token numbers.
