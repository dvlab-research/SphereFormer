import numpy as np
import random
import SharedArray as SA

import torch

from .voxelize import voxelize, voxelize_with_rec_idx, voxelize_with_rec_idx_v2


def sa_create(name, var):
    x = SA.create(name, var.shape, dtype=var.dtype)
    x[...] = var[...]
    x.flags.writeable = False
    return x


def collate_fn_limit_mix3d(batch, max_batch_points, logger, p):
    coord, feat, label = list(zip(*batch))
    offset, count = [], 0
    # print("coord:", len(coord))
    k = 0
    for item in coord:
        # print("item shape:",item.shape)
        count += item.shape[0]
        if count > max_batch_points:
            break
        k += 1
        offset.append(count)

    if logger is not None and k < len(batch):
        s = sum([x.shape[0] for x in coord])
        s_now = sum([x.shape[0] for x in coord[:k]])
        logger.warning("batch_size shortened from {} to {}, points from {} to {}".format(len(batch), k, s, s_now))

    if random.random() <= p:
        coord_mix3d = []
        feat_mix3d = []
        label_mix3d = []
        offset_mix3d = []
        for i in range(0, k, 2):
            if i == k-1:
                coord_mix3d_i = coord[i]
                feat_mix3d_i = feat[i]
                label_mix3d_i = label[i]
                offset_mix3d.append(offset[i])
            else:
                coord_mix3d_i = torch.cat([coord[i], coord[i+1]], 0)
                feat_mix3d_i = torch.cat([feat[i], feat[i+1]], 0)
                label_mix3d_i = torch.cat([label[i], label[i+1]], 0)
                offset_mix3d.append(offset[i+1])
            coord_mix3d.append(coord_mix3d_i)
            feat_mix3d.append(feat_mix3d_i)
            label_mix3d.append(label_mix3d_i)
        return torch.cat(coord_mix3d), torch.cat(feat_mix3d), torch.cat(label_mix3d), torch.IntTensor(offset_mix3d)

    return torch.cat(coord[:k]), torch.cat(feat[:k]), torch.cat(label[:k]), torch.IntTensor(offset[:k])
    # return torch.cat(coord), torch.cat(feat), torch.cat(label), torch.IntTensor(offset)
    # return torch.cat(coord_mix3d), torch.cat(feat_mix3d), torch.cat(label_mix3d), torch.IntTensor(offset_mix3d)


def collate_fn_limit(batch, max_batch_points, logger):
    coord, feat, label = list(zip(*batch))
    offset, count = [], 0
    # print("coord:", len(coord))
    k = 0
    for item in coord:
        # print("item shape:",item.shape)
        count += item.shape[0]
        if count > max_batch_points:
            break
        k += 1
        offset.append(count)

    if logger is not None and k < len(batch):
        s = sum([x.shape[0] for x in coord])
        s_now = sum([x.shape[0] for x in coord[:k]])
        logger.warning("batch_size shortened from {} to {}, points from {} to {}".format(len(batch), k, s, s_now))

    return torch.cat(coord[:k]), torch.cat(feat[:k]), torch.cat(label[:k]), torch.IntTensor(offset[:k])
    # return torch.cat(coord), torch.cat(feat), torch.cat(label), torch.IntTensor(offset)

def cart2sphere(xyz):
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    theta = (torch.atan2(y, x) + np.pi) * 180 / np.pi
    beta = torch.atan2(torch.sqrt(x**2 + y**2), z) * 180 / np.pi
    r = torch.sqrt(x**2 + y**2 + z**2)
    return torch.stack([theta, beta, r], -1)

def cart2polar(xyz):
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    theta = (torch.atan2(y, x) + np.pi) * 180 / np.pi
    r = torch.sqrt(x**2 + y**2)
    return torch.stack([theta, r, z], -1)

def collate_fn_limit_mix(batch, max_batch_points, logger, filter_type='none', args=None):
    coord, xyz, feat, label = list(zip(*batch))
    offset, count = [], 0
    
    new_coord, new_xyz, new_feat, new_label = [], [], [], []

    filtered = False
    k = 0
    for i, item in enumerate(xyz):
        # print("item shape:",item.shape)

        if filtered == False:
            if filter_type == 'circle':
            
                # print("xyz[i].shape: {}, xyz[i].min(0)[0]: {}, xyz[i].max(0)[0]: {}".format(xyz[i].shape, xyz[i].min(0)[0], xyz[i].max(0)[0]))

                pos_sphere = cart2sphere(xyz[i])
                
                pos_circle = pos_sphere.clone()
                t0, alpha, param_k = args.circle_scale_params
                bias = t0 ** (-1/alpha)

                # print("t0: {}, alpha: {}, bias: {}".format(t0, alpha, bias))

                pos_circle[:, 2] = - (param_k * pos_sphere[:, 2] + bias) ** (-alpha) + t0

                # print("e1 pos_circle.shape: {}, pos_circle.min(0)[0]: {}, pos_circle.max(0)[0]: {}".format(
                #     pos_circle.shape, pos_circle.min(0)[0], pos_circle.max(0)[0]
                # ))

                pos_circle = (pos_circle - pos_circle.min(0)[0]) // torch.tensor(args.window_size_circle).float()
                
                # print("e2 pos_circle.shape: {}, pos_circle.min(0)[0]: {}, pos_circle.max(0)[0]: {}".format(
                #     pos_circle.shape, pos_circle.min(0)[0], pos_circle.max(0)[0]
                # ))

                _, counts = torch.unique(pos_circle, return_counts=True, dim=0)

                # print("counts.min(): {}, counts.float().mean(): {}, counts.max(): {}".format(
                #     counts.min(), counts.float().mean(), counts.max()
                # ))

                if counts.max().item() > args.filter_threshold:
                    if logger is not None:
                        logger.warning("batch_size shortened by filter {}".format(filter_type))

                    filtered = True
                    continue
            elif filter_type == 'polar':
            
                # print("xyz[i].shape: {}, xyz[i].min(0)[0]: {}, xyz[i].max(0)[0]: {}".format(xyz[i].shape, xyz[i].min(0)[0], xyz[i].max(0)[0]))

                pos_sphere = cart2polar(xyz[i])
                
                # pos_circle = pos_sphere.clone()
                # t0, alpha, param_k = args.circle_scale_params
                # bias = t0 ** (-1/alpha)

                # print("t0: {}, alpha: {}, bias: {}".format(t0, alpha, bias))

                # pos_circle[:, 2] = - (param_k * pos_sphere[:, 2] + bias) ** (-alpha) + t0

                # print("e1 pos_sphere.shape: {}, pos_sphere.min(0)[0]: {}, pos_sphere.max(0)[0]: {}".format(
                #     pos_sphere.shape, pos_sphere.min(0)[0], pos_sphere.max(0)[0]
                # ))

                pos_sphere = (pos_sphere - pos_sphere.min(0)[0]) // torch.tensor(args.window_size_sphere).float()
                
                # print("e2 pos_sphere.shape: {}, pos_sphere.min(0)[0]: {}, pos_sphere.max(0)[0]: {}".format(
                #     pos_sphere.shape, pos_sphere.min(0)[0], pos_sphere.max(0)[0]
                # ))

                _, counts = torch.unique(pos_sphere, return_counts=True, dim=0)

                # print("counts.min(): {}, counts.float().mean(): {}, counts.max(): {}".format(
                #     counts.min(), counts.float().mean(), counts.max()
                # ))

                if counts.max().item() > args.filter_threshold:
                    if logger is not None:
                        logger.warning("batch_size shortened by filter {}".format(filter_type))

                    filtered = True
                    continue


        count += item.shape[0]
        if count > max_batch_points:
            break

        k += 1
        offset.append(count)
        new_coord.append(coord[i])
        new_xyz.append(xyz[i])
        new_feat.append(feat[i])
        new_label.append(label[i])

    if logger is not None and k < len(batch):
        s = sum([x.shape[0] for x in xyz])
        s_now = sum([x.shape[0] for x in new_xyz[:k]])
        logger.warning("batch_size shortened from {} to {}, points from {} to {}".format(len(batch), k, s, s_now))

    # print("len(new_coord): {}, offset: {}".format(len(new_coord), offset))

    # return torch.cat(coord[:k]), torch.cat(xyz[:k]), torch.cat(feat[:k]), torch.cat(label[:k]), torch.IntTensor(offset[:k])
    # return torch.cat(coord), torch.cat(feat), torch.cat(label), torch.IntTensor(offset)
    return torch.cat(new_coord[:k]), torch.cat(new_xyz[:k]), torch.cat(new_feat[:k]), torch.cat(new_label[:k]), torch.IntTensor(offset[:k])
    
def collate_fn_mlp_before_voxel(batch):
    uniq_idx, coord, xyz, feat, label, inds_recons = list(zip(*batch))
    inds_recons = list(inds_recons)
    uniq_idx = list(uniq_idx)
    
    offset, count, count_xyz = [], 0, 0
    # print("coord:", len(coord))
    for i, item in enumerate(coord):
        # print("item shape:",item.shape)
        inds_recons[i] = count + inds_recons[i]

        # print("type(uniq_idx[i]): {}".format(type(uniq_idx[i])))
        # print("uniq_idx[i][0]: ", uniq_idx[i][0])

        # print("uniq_idx[i].shape: {}, ".format(uniq_idx[i].shape))
        # print("count_xyz: ", count_xyz)

        uniq_idx[i] = count_xyz + uniq_idx[i]
        count += item.shape[0]
        count_xyz += xyz[i].shape[0]

        # print('count: {}, count_xyz: {}'.format(count, count_xyz))

        offset.append(count)
    return torch.cat(uniq_idx), torch.cat(coord), torch.cat(xyz), torch.cat(feat), torch.cat(label), torch.IntTensor(offset), torch.cat(inds_recons)

def collate_fn_mlp_before_voxel_test(batch):
    uniq_idx, coord, xyz, feat, label, inds_recons, paths = list(zip(*batch))
    inds_recons = list(inds_recons)
    uniq_idx = list(uniq_idx)
    
    offset, count, count_xyz = [], 0, 0
    for i, item in enumerate(coord):
        inds_recons[i] = count + inds_recons[i]
        uniq_idx[i] = count_xyz + uniq_idx[i]
        count += item.shape[0]
        count_xyz += xyz[i].shape[0]
        offset.append(count)
    return torch.cat(uniq_idx), torch.cat(coord), torch.cat(xyz), torch.cat(feat), torch.cat(label), torch.IntTensor(offset), torch.cat(inds_recons), paths

def collate_fn_mlp_before_voxel_test_tta(batch_list):
    samples = []

    batch_list = list(zip(*batch_list))

    for batch in batch_list:

        uniq_idx, coord, xyz, feat, label, inds_recons, paths = list(zip(*batch))
        inds_recons = list(inds_recons)
        uniq_idx = list(uniq_idx)
        
        offset, count, count_xyz = [], 0, 0
        for i, item in enumerate(coord):
            inds_recons[i] = count + inds_recons[i]
            uniq_idx[i] = count_xyz + uniq_idx[i]
            count += item.shape[0]
            count_xyz += xyz[i].shape[0]
            offset.append(count)
        sample = (torch.cat(uniq_idx), torch.cat(coord), torch.cat(xyz), torch.cat(feat), torch.cat(label), torch.IntTensor(offset), torch.cat(inds_recons), paths)
        samples.append(sample)
    return samples

# def collate_fn_mlp_before_voxel_test(batch):
#     coord, xyz, feat, label, inds_recons, paths = list(zip(*batch))
#     inds_recons = list(inds_recons)
    
#     offset, count = [], 0
#     # print("coord:", len(coord))
#     for i, item in enumerate(coord):
#         # print("item shape:",item.shape)
#         inds_recons[i] = count + inds_recons[i]
#         count += item.shape[0]
#         offset.append(count)
#     return torch.cat(coord), torch.cat(xyz), torch.cat(feat), torch.cat(label), torch.IntTensor(offset), torch.cat(inds_recons), paths

def collate_fn_with_reproj_idx(batch):
    coord, feat, label, inds_recons = list(zip(*batch))
    inds_recons = list(inds_recons)
    
    offset, count = [], 0
    # print("coord:", len(coord))
    for i, item in enumerate(coord):
        # print("item shape:",item.shape)
        inds_recons[i] = count + inds_recons[i]
        count += item.shape[0]
        offset.append(count)
    return torch.cat(coord), torch.cat(feat), torch.cat(label), torch.IntTensor(offset), torch.cat(inds_recons)

def collate_fn(batch):
    coord, feat, label = list(zip(*batch))
    offset, count = [], 0
    # print("coord:", len(coord))
    for item in coord:
        # print("item shape:",item.shape)
        count += item.shape[0]
        offset.append(count)
    return torch.cat(coord), torch.cat(feat), torch.cat(label), torch.IntTensor(offset)


def area_crop(coord, area_rate, split='train'):
    coord_min, coord_max = np.min(coord, 0), np.max(coord, 0)
    coord -= coord_min; coord_max -= coord_min
    x_max, y_max = coord_max[0:2]
    x_size, y_size = np.sqrt(area_rate) * x_max, np.sqrt(area_rate) * y_max
    if split == 'train' or split == 'trainval':
        x_s, y_s = random.uniform(0, x_max - x_size), random.uniform(0, y_max - y_size)
    else:
        x_s, y_s = (x_max - x_size) / 2, (y_max - y_size) / 2
    x_e, y_e = x_s + x_size, y_s + y_size
    crop_idx = np.where((coord[:, 0] >= x_s) & (coord[:, 0] <= x_e) & (coord[:, 1] >= y_s) & (coord[:, 1] <= y_e))[0]
    return crop_idx


def load_kitti_data(data_path):
    data = np.fromfile(data_path, dtype=np.float32)
    data = data.reshape((-1, 4))  # xyz+remission
    return data


def load_kitti_label(label_path, remap_lut):
    label = np.fromfile(label_path, dtype=np.uint32)
    label = label.reshape(-1)
    sem_label = label & 0xFFFF  # semantic label in lower half
    inst_label = label >> 16  # instance id in upper half
    assert ((sem_label + (inst_label << 16) == label).all())
    sem_label = remap_lut[sem_label]
    return sem_label.astype(np.int32)


def data_prepare(coord, feat, label, split='train', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False):
    if transform:
        coord, feat, label = transform(coord, feat, label)
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
    if voxel_max and label.shape[0] > voxel_max:
        init_idx = np.random.randint(label.shape[0]) if 'train' in split else label.shape[0] // 2
        crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
        coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

    coord_min, coord_max = np.min(coord, 0), np.max(coord, 0)
    coord -= (coord_min + coord_max) / 2.0
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat) / 255.
    label = torch.LongTensor(label)
    return coord, feat, label


def data_prepare_v101(coord, feat, label, split='train', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False):
    if transform:
        # coord, feat, label = transform(coord, feat, label)
        coord, feat = transform(coord, feat)
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
    if voxel_max and label.shape[0] > voxel_max:
        init_idx = np.random.randint(label.shape[0]) if 'train' in split else label.shape[0] // 2
        crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
        coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

    coord_min = np.min(coord, 0)
    coord -= coord_min
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat) / 255.
    label = torch.LongTensor(label)
    return coord, feat, label


def data_prepare_scannet(coord, feat, label, split='train', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False):
    if transform:
        # coord, feat, label = transform(coord, feat, label)
        coord, feat = transform(coord, feat)
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
    if voxel_max and label.shape[0] > voxel_max:
        init_idx = np.random.randint(label.shape[0]) if 'train' in split else label.shape[0] // 2
        crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
        coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

    coord_min = np.min(coord, 0)
    coord -= coord_min
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat)
    label = torch.LongTensor(label)
    return coord, feat, label


def data_prepare_nuscenes(coord, feat, label, split='train', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False, xyz_norm=True):
    if transform:
        # coord, feat, label = transform(coord, feat, label)
        coord, feat = transform(coord, feat)
    if voxel_size is not None:
        coord_min = np.min(coord, 0)
        # coord -= coord_min
        coord_norm = coord - coord_min
        if split == 'train':
            uniq_idx = voxelize_with_rec_idx(coord_norm, voxel_size)
        else:
            uniq_idx, idx_recon = voxelize_with_rec_idx(coord_norm, voxel_size, mode=1)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
    if voxel_max and label.shape[0] > voxel_max:
        init_idx = np.random.randint(label.shape[0]) if 'train' in split else label.shape[0] // 2
        crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
        coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
    # if shuffle_index:
    #     shuf_idx = np.arange(coord.shape[0])
    #     np.random.shuffle(shuf_idx)
    #     coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

    if xyz_norm:
        coord_min = np.min(coord, 0)
        coord -= coord_min
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat)
    label = torch.LongTensor(label)
    if split != 'train':
        idx_recon = torch.LongTensor(idx_recon)
        return coord, feat, label, idx_recon
    return coord, feat, label

def data_prepare_nuscenes_mix(coord, feat, label, split='train', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False, xyz_norm=True):
    if transform:
        # coord, feat, label = transform(coord, feat, label)
        coord, feat = transform(coord, feat)
    if voxel_size is not None:
        coord_min = np.min(coord, 0)
        # coord -= coord_min
        coord_norm = coord - coord_min
        if split == 'train':
            uniq_idx = voxelize_with_rec_idx(coord_norm, voxel_size)
        else:
            uniq_idx, idx_recon = voxelize_with_rec_idx(coord_norm, voxel_size, mode=1)
        coord_voxel = np.floor(coord_norm[uniq_idx] / np.array(voxel_size))
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
    if voxel_max and label.shape[0] > voxel_max:
        init_idx = np.random.randint(label.shape[0]) if 'train' in split else label.shape[0] // 2
        crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
        coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
        coord_voxel = coord_voxel[crop_idx]
    # if shuffle_index:
    #     shuf_idx = np.arange(coord.shape[0])
    #     np.random.shuffle(shuf_idx)
    #     coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

    if xyz_norm:
        coord_min = np.min(coord, 0)
        coord -= coord_min
    coord_voxel = torch.LongTensor(coord_voxel)
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat)
    label = torch.LongTensor(label)
    if split != 'train':
        idx_recon = torch.LongTensor(idx_recon)
        return coord_voxel, coord, feat, label, idx_recon
    return coord_voxel, coord, feat, label

def data_prepare_nuscenes_recon_idx(coord, split='train', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False, xyz_norm=True):
    # if transform:
    #     # coord, feat, label = transform(coord, feat, label)
    #     coord, feat = transform(coord, feat)
    if voxel_size is not None:
        coord_min = np.min(coord, 0)
        # coord -= coord_min
        coord_norm = coord - coord_min
        uniq_idx, idx_recon = voxelize_with_rec_idx_v2(coord_norm, voxel_size)

        coord_voxel = np.floor(coord_norm[uniq_idx] / np.array(voxel_size))
        # coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
    # if voxel_max and coord.shape[0] > voxel_max:
    #     init_idx = np.random.randint(label.shape[0]) if 'train' in split else label.shape[0] // 2
    #     crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
    #     coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
    #     coord_voxel = coord_voxel[crop_idx]
    # if shuffle_index:
    #     shuf_idx = np.arange(coord.shape[0])
    #     np.random.shuffle(shuf_idx)
    #     coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

    if xyz_norm:
        coord_min = np.min(coord, 0)
        coord = coord - coord_min
    coord_voxel = torch.LongTensor(coord_voxel)
    # coord = torch.FloatTensor(coord)
    # feat = torch.FloatTensor(feat)
    # label = torch.LongTensor(label)
    
    idx_recon = torch.LongTensor(idx_recon)
    # return coord_voxel, coord, feat, label, idx_recon
    return uniq_idx, coord_voxel, idx_recon

def data_prepare_nuscenes_meanvoxel(coord, feat, label, split='train', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False, xyz_norm=True):
    if transform:
        # coord, feat, label = transform(coord, feat, label)
        coord, feat = transform(coord, feat)
    if voxel_size is not None:
        coord_min = np.min(coord, 0)
        # coord -= coord_min
        coord_norm = coord - coord_min
        # if split == 'train':
        #     uniq_idx = voxelize_with_rec_idx(coord_norm, voxel_size)
        # else:
        #     uniq_idx, idx_recon = voxelize_with_rec_idx(coord_norm, voxel_size, mode=1)
        # coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
        uniq_idx, idx_recon = voxelize_with_rec_idx(coord_norm, voxel_size, mode=1)

    if voxel_max and label.shape[0] > voxel_max:
        init_idx = np.random.randint(label.shape[0]) if 'train' in split else label.shape[0] // 2
        crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
        coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
    # if shuffle_index:
    #     shuf_idx = np.arange(coord.shape[0])
    #     np.random.shuffle(shuf_idx)
    #     coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

    if xyz_norm:
        coord_min = np.min(coord, 0)
        coord -= coord_min
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat)
    label = torch.LongTensor(label)
    idx_recon = torch.LongTensor(idx_recon)
    # if split != 'train':
    #     idx_recon = torch.LongTensor(idx_recon)
    #     return coord, feat, label, idx_recon
    # return coord, feat, label
    return coord, feat, label, idx_recon


def data_prepare_v102(coord, feat, label, split='train', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False):
    if transform:
        coord, feat, label = transform(coord, feat, label)
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
    while voxel_max and label.shape[0] > voxel_max * 1.1:
        area_rate = voxel_max / float(label.shape[0])
        coord_min, coord_max = np.min(coord, 0), np.max(coord, 0)
        coord -= coord_min; coord_max -= coord_min
        x_max, y_max = coord_max[0:2]
        x_size, y_size = np.sqrt(area_rate) * x_max, np.sqrt(area_rate) * y_max
        if split == 'train':
            x_s, y_s = random.uniform(0, x_max - x_size), random.uniform(0, y_max - y_size)
        else:
            x_s, y_s = 0, 0
        x_e, y_e = x_s + x_size, y_s + y_size
        crop_idx = np.where((coord[:, 0] >= x_s) & (coord[:, 0] <= x_e) & (coord[:, 1] >= y_s) & (coord[:, 1] <= y_e))[0]
        if crop_idx.shape[0] < voxel_max // 8: continue
        coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]

    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

    coord_min = np.min(coord, 0)
    coord -= coord_min
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat) / 255.
    label = torch.LongTensor(label)
    return coord, feat, label


def data_prepare_v103(coord, feat, label, split='train', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False):
    if transform:
        coord, feat, label = transform(coord, feat, label)
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
    if voxel_max and label.shape[0] > voxel_max:
        coord_min, coord_max = np.min(coord, 0), np.max(coord, 0)
        coord -= coord_min; coord_max -= coord_min
        xy_area = 7
        while True:
            x_area, y_area = np.random.randint(xy_area), np.random.randint(xy_area)
            x_s, y_s = coord_max[0] * x_area / float(xy_area), coord_max[1] * y_area / float(xy_area)
            x_e, y_e = coord_max[0] * (x_area + 1) / float(xy_area), coord_max[1] * (y_area + 1) / float(xy_area)
            crop_idx = np.where((coord[:, 0] >= x_s) & (coord[:, 0] <= x_e) & (coord[:, 1] >= y_s) & (coord[:, 1] <= y_e))[0]
            if crop_idx.shape[0] > 0:
                init_idx = crop_idx[np.random.randint(crop_idx.shape[0])] if 'train' in split else label.shape[0] // 2
                crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
                coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
                break
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

    coord_min = np.min(coord, 0)
    coord -= coord_min
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat) / 255.
    label = torch.LongTensor(label)
    return coord, feat, label


def data_prepare_v104(coord, feat, label, split='train', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False):
    if transform:
        coord, feat, label = transform(coord, feat, label)
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
    if voxel_max and label.shape[0] > voxel_max:
        coord_min, coord_max = np.min(coord, 0), np.max(coord, 0)
        coord -= coord_min; coord_max -= coord_min
        xy_area = 10
        while True:
            x_area, y_area = np.random.randint(xy_area), np.random.randint(xy_area)
            x_s, y_s = coord_max[0] * x_area / float(xy_area), coord_max[1] * y_area / float(xy_area)
            x_e, y_e = coord_max[0] * (x_area + 1) / float(xy_area), coord_max[1] * (y_area + 1) / float(xy_area)
            crop_idx = np.where((coord[:, 0] >= x_s) & (coord[:, 0] <= x_e) & (coord[:, 1] >= y_s) & (coord[:, 1] <= y_e))[0]
            if crop_idx.shape[0] > 0:
                init_idx = crop_idx[np.random.randint(crop_idx.shape[0])] if 'train' in split else label.shape[0] // 2
                crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
                coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
                break
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

    coord_min = np.min(coord, 0)
    coord -= coord_min
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat) / 255.
    label = torch.LongTensor(label)
    return coord, feat, label


def data_prepare_v105(coord, feat, label, split='train', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False):
    if transform:
        coord, feat, label = transform(coord, feat, label)
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
    if voxel_max and label.shape[0] > voxel_max:
        init_idx = np.random.randint(label.shape[0]) if 'train' in split else label.shape[0] // 2
        crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
        coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

    coord_min = np.min(coord, 0)
    coord[:, 0:2] -= coord_min[0:2]
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat) / 255.
    label = torch.LongTensor(label)
    return coord, feat, label
