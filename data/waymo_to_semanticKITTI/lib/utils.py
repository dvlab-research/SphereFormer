import argparse
import tensorflow.compat.v1 as tf
import numpy as np

tf.enable_eager_execution()

from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.protos import segmentation_metrics_pb2
from waymo_open_dataset.protos import segmentation_submission_pb2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_dir", help="load_dir", default='[YOUR_DARA_ROOT]/waymo_open_dataset')
    parser.add_argument("--save_dir", help="save_dir", default='[YOUR_DARA_ROOT]/semantickitti_format')
    return parser.parse_args()


def convert_range_image_to_point_cloud_labels(frame,
                                              range_images,
                                              segmentation_labels,
                                              ri_index=0):
    """Convert segmentation labels from range images to point clouds.

    Args:
    frame: open dataset frame
    range_images: A dict of {laser_name, [range_image_first_return,
    range_image_second_return]}.
    segmentation_labels: A dict of {laser_name, [range_image_first_return,
    range_image_second_return]}.
    ri_index: 0 for the first return, 1 for the second return.

    Returns:
    point_labels: {[N, 2]} list of 3d lidar points's segmentation labels. 0 for
    points that are not labeled.
    """
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    point_labels = []
    for c in calibrations:
        range_image = range_images[c.name][ri_index]
        range_image_tensor = tf.reshape(
            tf.convert_to_tensor(range_image.data), range_image.shape.dims)
        range_image_mask = range_image_tensor[..., 0] > 0

        if c.name in segmentation_labels:
            sl = segmentation_labels[c.name][ri_index]
            sl_tensor = tf.reshape(tf.convert_to_tensor(sl.data), sl.shape.dims)
            sl_points_tensor = tf.gather_nd(sl_tensor, tf.where(range_image_mask))
        else:
            num_valid_point = tf.math.reduce_sum(tf.cast(range_image_mask, tf.int32))
            sl_points_tensor = tf.zeros([num_valid_point, 2], dtype=tf.int32)

        point_labels.append(sl_points_tensor.numpy())
    return point_labels


def convert_range_image_to_point_cloud(frame,
                                        range_images,
                                        camera_projections,
                                        range_image_top_pose,
                                        ri_index=0):
    """Convert range images to point cloud.
    Args:
        frame (:obj:`Frame`): Open dataset frame.
        range_images (dict): Mapping from laser_name to list of two
            range images corresponding with two returns.
        camera_projections (dict): Mapping from laser_name to list of two
            camera projections corresponding with two returns.
        range_image_top_pose (:obj:`Transform`): Range image pixel pose for
            top lidar.
        ri_index (int, optional): 0 for the first return,
            1 for the second return. Default: 0.
    Returns:
        tuple[list[np.ndarray]]: (List of points with shape [N, 3],
            camera projections of points with shape [N, 6], intensity
            with shape [N, 1], elongation with shape [N, 1], points'
            position in the depth map (element offset if points come from
            the main lidar otherwise -1) with shape[N, 1]). All the
            lists have the length of lidar numbers (5).
    """
    calibrations = sorted(
        frame.context.laser_calibrations, key=lambda c: c.name)
    points = []
    cp_points = []
    intensity = []
    elongation = []
    mask_indices = []

    frame_pose = tf.convert_to_tensor(
        value=np.reshape(np.array(frame.pose.transform), [4, 4]))
    # [H, W, 6]
    range_image_top_pose_tensor = tf.reshape(
        tf.convert_to_tensor(value=range_image_top_pose.data),
        range_image_top_pose.shape.dims)
    # [H, W, 3, 3]
    range_image_top_pose_tensor_rotation = \
        transform_utils.get_rotation_matrix(
            range_image_top_pose_tensor[..., 0],
            range_image_top_pose_tensor[..., 1],
            range_image_top_pose_tensor[..., 2])
    range_image_top_pose_tensor_translation = \
        range_image_top_pose_tensor[..., 3:]
    range_image_top_pose_tensor = transform_utils.get_transform(
        range_image_top_pose_tensor_rotation,
        range_image_top_pose_tensor_translation)
    for c in calibrations:
        range_image = range_images[c.name][ri_index]
        if len(c.beam_inclinations) == 0:
            beam_inclinations = range_image_utils.compute_inclination(
                tf.constant(
                    [c.beam_inclination_min, c.beam_inclination_max]),
                height=range_image.shape.dims[0])
        else:
            beam_inclinations = tf.constant(c.beam_inclinations)

        beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
        extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

        range_image_tensor = tf.reshape(
            tf.convert_to_tensor(value=range_image.data),
            range_image.shape.dims)
        pixel_pose_local = None
        frame_pose_local = None
        if c.name == dataset_pb2.LaserName.TOP:
            pixel_pose_local = range_image_top_pose_tensor
            pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
            frame_pose_local = tf.expand_dims(frame_pose, axis=0)
        range_image_mask = range_image_tensor[..., 0] > 0

        nlz_mask = range_image_tensor[..., 3] != 1.0  # 1.0: in NLZ
        range_image_mask = range_image_mask & nlz_mask

        range_image_cartesian = \
            range_image_utils.extract_point_cloud_from_range_image(
                tf.expand_dims(range_image_tensor[..., 0], axis=0),
                tf.expand_dims(extrinsic, axis=0),
                tf.expand_dims(tf.convert_to_tensor(
                    value=beam_inclinations), axis=0),
                pixel_pose=pixel_pose_local,
                frame_pose=frame_pose_local)

        mask_index = tf.where(range_image_mask)

        range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
        points_tensor = tf.gather_nd(range_image_cartesian, mask_index)

        cp = camera_projections[c.name][ri_index]
        cp_tensor = tf.reshape(
            tf.convert_to_tensor(value=cp.data), cp.shape.dims)
        cp_points_tensor = tf.gather_nd(cp_tensor, mask_index)
        points.append(points_tensor.numpy())
        cp_points.append(cp_points_tensor.numpy())

        intensity_tensor = tf.gather_nd(range_image_tensor[..., 1],
                                        mask_index)
        intensity.append(intensity_tensor.numpy())

        elongation_tensor = tf.gather_nd(range_image_tensor[..., 2],
                                            mask_index)
        elongation.append(elongation_tensor.numpy())
        if c.name == 1:
            mask_index = (ri_index * range_image_mask.shape[0] +
                            mask_index[:, 0]
                            ) * range_image_mask.shape[1] + mask_index[:, 1]
            mask_index = mask_index.numpy().astype(elongation[-1].dtype)
        else:
            mask_index = np.full_like(elongation[-1], -1)

        mask_indices.append(mask_index)

    return points, cp_points, intensity, elongation, mask_indices