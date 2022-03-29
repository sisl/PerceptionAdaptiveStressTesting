import math
import os
import os.path as osp
from typing import List
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from nuscenes import NuScenes
import nuscenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.eval.common.data_classes import EvalBox, EvalBoxes
from nuscenes.eval.common.utils import boxes_to_sensor
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.eval.tracking.data_classes import TrackingBox
from nuscenes.eval.tracking.utils import category_to_tracking_name
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import BoxVisibility, view_points
from pyquaternion.quaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from PIL import Image


def load_sample_gt(nusc: NuScenes, sample_token: str, box_cls, verbose: bool = False) -> EvalBoxes:
    """
    Loads ground truth boxes from DB.
    :param nusc: A NuScenes instance.
    :param eval_split: The evaluation split for which we load GT boxes.
    :param box_cls: Type of box to load, e.g. DetectionBox or TrackingBox.
    :param verbose: Whether to print messages to stdout.
    :return: The GT boxes.
    """
    # Init.
    if box_cls == DetectionBox:
        attribute_map = {a['token']: a['name'] for a in nusc.attribute}
    version = nusc.version


    all_annotations = EvalBoxes()

    # Load annotations and filter predictions and annotations.
    tracking_id_set = set()

    sample = nusc.get('sample', sample_token)
    sample_annotation_tokens = sample['anns']

    sample_boxes = []
    for sample_annotation_token in sample_annotation_tokens:

        sample_annotation = nusc.get('sample_annotation', sample_annotation_token)
        if box_cls == DetectionBox:
            # Get label name in detection task and filter unused labels.
            detection_name = category_to_detection_name(sample_annotation['category_name'])
            if detection_name is None:
                continue

            # Get attribute_name.
            attr_tokens = sample_annotation['attribute_tokens']
            attr_count = len(attr_tokens)
            if attr_count == 0:
                attribute_name = ''
            elif attr_count == 1:
                attribute_name = attribute_map[attr_tokens[0]]
            else:
                raise Exception('Error: GT annotations must not have more than one attribute!')

            sample_boxes.append(
                box_cls(
                    sample_token=sample_token,
                    translation=sample_annotation['translation'],
                    size=sample_annotation['size'],
                    rotation=sample_annotation['rotation'],
                    velocity=nusc.box_velocity(sample_annotation['token'])[:2],
                    num_pts=sample_annotation['num_lidar_pts'] + sample_annotation['num_radar_pts'],
                    detection_name=detection_name,
                    detection_score=-1.0,  # GT samples do not have a score.
                    attribute_name=attribute_name
                )
            )
        elif box_cls == TrackingBox:
            # Use nuScenes token as tracking id.
            tracking_id = sample_annotation['instance_token']
            tracking_id_set.add(tracking_id)

            # Get label name in detection task and filter unused labels.
            # Import locally to avoid errors when motmetrics package is not installed.
            
            tracking_name = category_to_tracking_name(sample_annotation['category_name'])
            if tracking_name is None:
                continue

            sample_boxes.append(
                box_cls(
                    sample_token=sample_token,
                    translation=sample_annotation['translation'],
                    size=sample_annotation['size'],
                    rotation=sample_annotation['rotation'],
                    velocity=nusc.box_velocity(sample_annotation['token'])[:2],
                    num_pts=sample_annotation['num_lidar_pts'] + sample_annotation['num_radar_pts'],
                    tracking_id=tracking_id,
                    tracking_name=tracking_name,
                    tracking_score=-1.0  # GT samples do not have a score.
                )
            )
        else:
            raise NotImplementedError('Error: Invalid box_cls %s!' % box_cls)

    all_annotations.add_boxes(sample_token, sample_boxes)

    if verbose:
        print("Loaded ground truth annotations for {} samples.".format(len(all_annotations.sample_tokens)))

    return all_annotations


def get_boxes_for_annotation(nusc: NuScenes, sample_data_token: str, annotation_token: str) -> List[Box]:
    """
    Instantiates Boxes for all annotation for a particular sample_data record. If the sample_data is a
    keyframe, this returns the annotations for that sample. But if the sample_data is an intermediate
    sample_data, a linear interpolation is applied to estimate the location of the boxes at the time the
    sample_data was captured.
    :param sample_data_token: Unique sample_data identifier.
    """

    # Retrieve sensor & pose records
    sd_record = nusc.get('sample_data', sample_data_token)
    curr_sample_record = nusc.get('sample', sd_record['sample_token'])
    curr_ann_record = nusc.get('sample_annotation', annotation_token)

    if curr_sample_record['prev'] == "" or sd_record['is_key_frame'] or curr_ann_record['prev'] == "":
        # If no previous annotations available, or if sample_data is keyframe just return the current ones.
        # boxes = list(map(self.get_box, curr_sample_record['anns']))
        boxes = [nusc.get_box(annotation_token)]

    else:
        prev_sample_record = nusc.get('sample', curr_sample_record['prev'])

        # curr_ann_recs = [nusc.get('sample_annotation', token) for token in curr_sample_record['anns']]
        # prev_ann_recs = [nusc.get('sample_annotation', token) for token in prev_sample_record['anns']]
        curr_ann_recs = [nusc.get('sample_annotation', annotation_token)]
        prev_ann_recs = [nusc.get('sample_annotation', curr_ann_record['prev'])]

        # Maps instance tokens to prev_ann records
        prev_inst_map = {entry['instance_token']: entry for entry in prev_ann_recs}

        t0 = prev_sample_record['timestamp']
        t1 = curr_sample_record['timestamp']
        t = sd_record['timestamp']

        # There are rare situations where the timestamps in the DB are off so ensure that t0 < t < t1.
        t = max(t0, min(t1, t))

        boxes = []
        for curr_ann_rec in curr_ann_recs:

            if curr_ann_rec['instance_token'] in prev_inst_map:
                # If the annotated instance existed in the previous frame, interpolate center & orientation.
                prev_ann_rec = prev_inst_map[curr_ann_rec['instance_token']]

                # Interpolate center.
                center = [np.interp(t, [t0, t1], [c0, c1]) for c0, c1 in zip(prev_ann_rec['translation'],
                                                                                curr_ann_rec['translation'])]

                # Interpolate orientation.
                rotation = Quaternion.slerp(q0=Quaternion(prev_ann_rec['rotation']),
                                            q1=Quaternion(curr_ann_rec['rotation']),
                                            amount=(t - t0) / (t1 - t0))

                box = Box(center, curr_ann_rec['size'], rotation, name=curr_ann_rec['category_name'],
                            token=curr_ann_rec['token'])
            else:
                # If not, simply grab the current annotation.
                box = nusc.get_box(curr_ann_rec['token'])

            boxes.append(box)
    return boxes[0]


def get_ordered_samples(nuscenes: NuScenes, scene_token: str):
    """
    Get all sample tokens for scene in order
    """
    scene = nuscenes.get('scene', scene_token)
    scene_start_token = scene['first_sample_token']
    next_sample_token = nuscenes.get('sample', scene_start_token)['next']
    ordered_samples = [scene_start_token]
    while next_sample_token != '':
        ordered_samples.append(next_sample_token)
        next_sample_token = nuscenes.get('sample', next_sample_token)['next']

    return ordered_samples


def vis_sample_pointcloud(nusc: NuScenes,
                     sample_token: str,
                     gt_boxes: EvalBoxes,
                     pred_boxes: EvalBoxes,
                     pc: np.ndarray,
                     conf_th: float = 0.00,
                     eval_range: float = 50,
                     verbose: bool = True,
                     savepath: str = None) -> None:
    """
    Visualizes a sample from BEV with annotations and detection results.
    :param nusc: NuScenes object.
    :param sample_token: The nuScenes sample token.
    :param gt_boxes: Ground truth boxes grouped by sample.
    :param pred_boxes: Prediction grouped by sample.
    :param pc: Lidar point cloud for sample.
    :param conf_th: The confidence threshold used to filter negatives.
    :param eval_range: Range in meters beyond which boxes are ignored.
    :param verbose: Whether to print to stdout.
    :param savepath: If given, saves the the rendering here instead of displaying.
    """
    # Retrieve sensor & pose records.
    sample_rec = nusc.get('sample', sample_token)
    sd_record = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    # Get boxes.
    boxes_gt_global = gt_boxes[sample_token]
    boxes_est_global = pred_boxes[sample_token]

    # Map GT boxes to lidar.
    boxes_gt = boxes_to_sensor(boxes_gt_global, pose_record, cs_record)

    # Map EST boxes to lidar.
    boxes_est = boxes_to_sensor(boxes_est_global, pose_record, cs_record)

    # Add scores to EST boxes.
    for box_est, box_est_global in zip(boxes_est, boxes_est_global):
        box_est.score = box_est_global.detection_score

    # Get point cloud in lidar frame.
    #pc, _ = LidarPointCloud.from_file_multisweep(nusc, sample_rec, 'LIDAR_TOP', 'LIDAR_TOP', nsweeps=nsweeps)

    # Init axes.
    _, ax = plt.subplots(1, 1, figsize=(9, 9))

    render_ego_centric_map_alt(nusc, sample_data_token=sd_record['token'], axes_limit=eval_range, ax=ax)

    # Show point cloud.
    points = view_points(pc[:3, :], np.eye(4), normalize=False)
    dists = np.sqrt(np.sum(pc[:2, :] ** 2, axis=0))
    colors = np.minimum(1, dists / eval_range)
    ax.scatter(points[0, :], points[1, :], c='gray', s=0.2)

    # Show ego vehicle.
    ax.plot(0, 0, 'x', color='black')

    # Show GT boxes.
    for box in boxes_gt:
        box.render(ax, view=np.eye(4), colors=('k', 'k', 'k'), linewidth=2)

    # Show EST boxes.
    for box in boxes_est:
        # Show only predictions with a high score.
        assert not np.isnan(box.score), 'Error: Box score cannot be NaN!'
        if box.score >= conf_th:
            box.render(ax, view=np.eye(4), colors=('b', 'b', 'b'), linewidth=2)

    # Limit visible range.
    axes_limit = eval_range  # Slightly bigger to include boxes that extend beyond the range.
    ax.set_xlim(-axes_limit, axes_limit)
    ax.set_ylim(-axes_limit, axes_limit)
    ax.axis('off')
    ax.set_aspect('equal')

    # Show / save plot.
    if verbose:
        print('Rendering sample token %s' % sample_token)
    plt.title(sample_token)
    if savepath is not None:
        plt.savefig(savepath, dpi=1000)
        plt.close()
    else:
        pass
        # plt.show()

    return ax


def render_box_with_pc(nusc: NuScenes,
                       sample_token: str,
                       instance_token: str,
                       points: np.ndarray,
                       pred_boxes: List[EvalBox] = None,
                       margin: float = 10,
                       view: np.ndarray = np.eye(4),
                       box_vis_level: BoxVisibility = BoxVisibility.ANY,
                       out_path: str = None,
                       extra_info: bool = False) -> None:

        
        #ann_record = self.nusc.get('sample_annotation', anntoken)
        sample_record = nusc.get('sample', sample_token)
        for anntoken in sample_record['anns']:
            ann_record = nusc.get('sample_annotation', anntoken)
            if ann_record['instance_token'] == instance_token:
                break

        sample_rec = nusc.get('sample', sample_token)
        sd_record = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
        #det_boxes = EvalBoxes.deserialize(detections['results'], DetectionBox)
        # gt_boxes = load_gt(self.nuscenes, 'mini_val', DetectionBox)
        # gt_boxes = load_sample_gt(nusc, sample_token, DetectionBox)
        # gt_boxes_list = boxes_to_sensor(gt_boxes[sample_token], pose_record, cs_record)
        gt_box = nusc.get_box(ann_record['token'])
        gt_box.translate(-np.array(pose_record['translation']))
        gt_box.rotate(Quaternion(pose_record['rotation']).inverse)

        #  Move box to sensor coord system.
        gt_box.translate(-np.array(cs_record['translation']))
        gt_box.rotate(Quaternion(cs_record['rotation']).inverse)

        # for box in gt_boxes_list:
        #     if box.token == ann_record['token']:
        #         gt_box = box
        #         break

        fig, ax = plt.subplots(1, 1, figsize=(9, 9))

        # Plot LIDAR view.
        lidar = sample_record['data']['LIDAR_TOP']
        #data_path, boxes, camera_intrinsic = nusc.get_sample_data(lidar, selected_anntokens=[anntoken])
        #LidarPointCloud.from_file(data_path).render_height(axes[0], view=view)
        #print(points.shape)
        #LidarPointCloud(points).render_height(ax, view=view)
        points = view_points(points[:3, :], view, normalize=False)
        ax.scatter(points[0, :], points[1, :], c='gray', s=1)
        x_lim = (-20, 20)
        y_lim = (-20, 20)
        ax.set_xlim(x_lim[0], x_lim[1])
        ax.set_ylim(y_lim[0], y_lim[1])
       
        # for box in boxes:
        #     c = np.array(nusc.get_color(box.name)) / 255.0
        #     box.render(axes[0], view=view, colors=(c, c, c))
        #     corners = view_points(boxes[0].corners(), view, False)[:2, :]
        #     axes[0].set_xlim([np.min(corners[0, :]) - margin, np.max(corners[0, :]) + margin])
        #     axes[0].set_ylim([np.min(corners[1, :]) - margin, np.max(corners[1, :]) + margin])
        #     axes[0].axis('off')
        #     axes[0].set_aspect('equal')
        #box = nusc.get_box(ann_record['token'])
        #c = np.array(nusc.get_color(box.name)) / 255.0
        gt_box.render(ax, view=view, colors=('k', 'k', 'k'), linewidth=2)
        corners = view_points(gt_box.corners(), view, False)[:2, :]
        ax.set_xlim([np.min(corners[0, :]) - margin, np.max(corners[0, :]) + margin])
        ax.set_ylim([np.min(corners[1, :]) - margin, np.max(corners[1, :]) + margin])
        ax.axis('off')
        ax.set_aspect('equal')

        rel_pred_boxes = []
        if pred_boxes is not None:
            rel_pred_boxes = boxes_to_sensor(pred_boxes, pose_record, cs_record)
        
        for i,box in enumerate(rel_pred_boxes):
            if pred_boxes[i].detection_score > 0.3:
                box.render(ax, view=view, colors=('b', 'b', 'b'), linewidth=2)


        return ax


def annotation_for_sample_instance(nusc: NuScenes, sample_token: str, instance_token: str):
    """
    Return annotation corresponding to instance in sample
    """

    ann_tokens = nusc.field2token('sample_annotation', 'instance_token', instance_token)
    
    ann = None
    for token in ann_tokens:
        temp_ann = nusc.get('sample_annotation', token)
        if temp_ann['sample_token'] == sample_token:
            ann = temp_ann

    return ann




def render_sample_data(nusc,
                        sample_data_token: str,
                        with_anns: bool = True,
                        box_vis_level: BoxVisibility = BoxVisibility.ANY,
                        axes_limit: float = 40,
                        ax: Axes = None,
                        nsweeps: int = 1,
                        out_path: str = None,
                        underlay_map: bool = True,
                        use_flat_vehicle_coordinates: bool = True,
                        show_lidarseg: bool = False,
                        show_lidarseg_legend: bool = False,
                        filter_lidarseg_labels: List = None,
                        lidarseg_preds_bin_path: str = None,
                        verbose: bool = True,
                        show_panoptic: bool = False,
                        point_cloud: LidarPointCloud = None,
                        det_boxes:List[DetectionBox] = None,
                        det_thresh = 0.25) -> None:
    """
    Render sample data onto axis.
    :param sample_data_token: Sample_data token.
    :param with_anns: Whether to draw box annotations.
    :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
    :param axes_limit: Axes limit for lidar and radar (measured in meters).
    :param ax: Axes onto which to render.
    :param nsweeps: Number of sweeps for lidar and radar.
    :param out_path: Optional path to save the rendered figure to disk.
    :param underlay_map: When set to true, lidar data is plotted onto the map. This can be slow.
    :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
        aligned to z-plane in the world. Note: Previously this method did not use flat vehicle coordinates, which
        can lead to small errors when the vertical axis of the global frame and lidar are not aligned. The new
        setting is more correct and rotates the plot by ~90 degrees.
    :param show_lidarseg: When set to True, the lidar data is colored with the segmentation labels. When set
        to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
    :param show_lidarseg_legend: Whether to display the legend for the lidarseg labels in the frame.
    :param filter_lidarseg_labels: Only show lidar points which belong to the given list of classes. If None
        or the list is empty, all classes will be displayed.
    :param lidarseg_preds_bin_path: A path to the .bin file which contains the user's lidar segmentation
                                    predictions for the sample.
    :param verbose: Whether to display the image after it is rendered.
    :param show_panoptic: When set to True, the lidar data is colored with the panoptic labels. When set
        to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
        If show_lidarseg is True, show_panoptic will be set to False.
    """
    if show_lidarseg:
        show_panoptic = False
    # Get sensor modality.
    sd_record = nusc.get('sample_data', sample_data_token)
    sensor_modality = sd_record['sensor_modality']

    if sensor_modality in ['lidar', 'radar']:
        sample_rec = nusc.get('sample', sd_record['sample_token'])
        chan = sd_record['channel']
        ref_chan = 'LIDAR_TOP'
        ref_sd_token = sample_rec['data'][ref_chan]
        ref_sd_record = nusc.get('sample_data', ref_sd_token)

        if sensor_modality == 'lidar':
            if show_lidarseg or show_panoptic:
                gt_from = 'lidarseg' if show_lidarseg else 'panoptic'
                assert hasattr(nusc, gt_from), f'Error: nuScenes-{gt_from} not installed!'

                # Ensure that lidar pointcloud is from a keyframe.
                assert sd_record['is_key_frame'], \
                    'Error: Only pointclouds which are keyframes have lidar segmentation labels. Rendering aborted.'

                assert nsweeps == 1, \
                    'Error: Only pointclouds which are keyframes have lidar segmentation labels; nsweeps should ' \
                    'be set to 1.'

                # Load a single lidar point cloud.
                pcl_path = osp.join(nusc.dataroot, ref_sd_record['filename'])
                pc = LidarPointCloud.from_file(pcl_path)
            else:
                # Get aggregated lidar point cloud in lidar frame.
                pc, times = LidarPointCloud.from_file_multisweep(nusc, sample_rec, chan, ref_chan,
                                                                    nsweeps=nsweeps)
                
            velocities = None
        else:
            # Get aggregated radar point cloud in reference frame.
            # The point cloud is transformed to the reference frame for visualization purposes.
            # pc, times = RadarPointCloud.from_file_multisweep(nusc, sample_rec, chan, ref_chan, nsweeps=nsweeps)

            # # Transform radar velocities (x is front, y is left), as these are not transformed when loading the
            # # point cloud.
            # radar_cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
            # ref_cs_record = nusc.get('calibrated_sensor', ref_sd_record['calibrated_sensor_token'])
            # velocities = pc.points[8:10, :]  # Compensated velocity
            # velocities = np.vstack((velocities, np.zeros(pc.points.shape[1])))
            # velocities = np.dot(Quaternion(radar_cs_record['rotation']).rotation_matrix, velocities)
            # velocities = np.dot(Quaternion(ref_cs_record['rotation']).rotation_matrix.T, velocities)
            # velocities[2, :] = np.zeros(pc.points.shape[1])
            pass

        # By default we render the sample_data top down in the sensor frame.
        # This is slightly inaccurate when rendering the map as the sensor frame may not be perfectly upright.
        # Using use_flat_vehicle_coordinates we can render the map in the ego frame instead.
        if point_cloud:
            pc = point_cloud
        if use_flat_vehicle_coordinates:
            # Retrieve transformation matrices for reference point cloud.
            cs_record = nusc.get('calibrated_sensor', ref_sd_record['calibrated_sensor_token'])
            pose_record = nusc.get('ego_pose', ref_sd_record['ego_pose_token'])
            ref_to_ego =transform_matrix(translation=cs_record['translation'],
                                            rotation=Quaternion(cs_record["rotation"]))

            # Compute rotation between 3D vehicle pose and "flat" vehicle pose (parallel to global z plane).
            ego_yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
            rotation_vehicle_flat_from_vehicle = np.dot(
                Quaternion(scalar=np.cos(ego_yaw / 2), vector=[0, 0, np.sin(ego_yaw / 2)]).rotation_matrix,
                Quaternion(pose_record['rotation']).inverse.rotation_matrix)
            vehicle_flat_from_vehicle = np.eye(4)
            vehicle_flat_from_vehicle[:3, :3] = rotation_vehicle_flat_from_vehicle
            viewpoint = np.dot(vehicle_flat_from_vehicle, ref_to_ego)
        else:
            viewpoint = np.eye(4)

        # Init axes.
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(9, 9))

        # Render map if requested.
        if underlay_map:
            assert use_flat_vehicle_coordinates, 'Error: underlay_map requires use_flat_vehicle_coordinates, as ' \
                                                    'otherwise the location does not correspond to the map!'
            # nusc.explorer.render_ego_centric_map(sample_data_token=sample_data_token, axes_limit=axes_limit, ax=ax)
            render_ego_centric_map(nusc, sample_data_token=sample_data_token, axes_limit=axes_limit, ax=ax)

        # Show point cloud.
        points = view_points(pc.points[:3, :], viewpoint, normalize=False)
        dists = np.sqrt(np.sum(pc.points[:2, :] ** 2, axis=0))
        colors = np.minimum(1, dists / axes_limit / np.sqrt(2))
        if sensor_modality == 'lidar' and (show_lidarseg or show_panoptic):
            gt_from = 'lidarseg' if show_lidarseg else 'panoptic'
            semantic_table = getattr(nusc, gt_from)
            # Load labels for pointcloud.
            if lidarseg_preds_bin_path:
                sample_token = nusc.get('sample_data', sample_data_token)['sample_token']
                lidarseg_labels_filename = lidarseg_preds_bin_path
                assert os.path.exists(lidarseg_labels_filename), \
                    'Error: Unable to find {} to load the predictions for sample token {} (lidar ' \
                    'sample data token {}) from.'.format(lidarseg_labels_filename, sample_token, sample_data_token)
            else:
                if len(semantic_table) > 0:
                    # Ensure {lidarseg/panoptic}.json is not empty (e.g. in case of v1.0-test).
                    lidarseg_labels_filename = osp.join(nusc.dataroot,
                                                        nusc.get(gt_from, sample_data_token)['filename'])
                else:
                    lidarseg_labels_filename = None

            # if lidarseg_labels_filename:
            #     # Paint each label in the pointcloud with a RGBA value.
            #     if show_lidarseg or show_panoptic:
            #         if show_lidarseg:
            #             colors = paint_points_label(lidarseg_labels_filename, filter_lidarseg_labels,
            #                                         nusc.lidarseg_name2idx_mapping, self.nusc.colormap)
            #         else:
            #             colors = paint_panop_points_label(lidarseg_labels_filename, filter_lidarseg_labels,
            #                                                 self.nusc.lidarseg_name2idx_mapping, self.nusc.colormap)

            #         if show_lidarseg_legend:

            #             # If user does not specify a filter, then set the filter to contain the classes present in
            #             # the pointcloud after it has been projected onto the image; this will allow displaying the
            #             # legend only for classes which are present in the image (instead of all the classes).
            #             if filter_lidarseg_labels is None:
            #                 if show_lidarseg:
            #                     # Since the labels are stored as class indices, we get the RGB colors from the
            #                     # colormap in an array where the position of the RGB color corresponds to the index
            #                     # of the class it represents.
            #                     color_legend = colormap_to_colors(self.nusc.colormap,
            #                                                         self.nusc.lidarseg_name2idx_mapping)
            #                     filter_lidarseg_labels = get_labels_in_coloring(color_legend, colors)
            #                 else:
            #                     # Only show legends for stuff categories for panoptic.
            #                     filter_lidarseg_labels = stuff_cat_ids(len(self.nusc.lidarseg_name2idx_mapping))

            #             if filter_lidarseg_labels and show_panoptic:
            #                 # Only show legends for filtered stuff categories for panoptic.
            #                 stuff_labels = set(stuff_cat_ids(len(self.nusc.lidarseg_name2idx_mapping)))
            #                 filter_lidarseg_labels = list(stuff_labels.intersection(set(filter_lidarseg_labels)))

            #             create_lidarseg_legend(filter_lidarseg_labels,
            #                                     self.nusc.lidarseg_idx2name_mapping,
            #                                     self.nusc.colormap,
            #                                     loc='upper left',
            #                                     ncol=1,
            #                                     bbox_to_anchor=(1.05, 1.0))
            # else:
                # print('Warning: There are no lidarseg labels in {}. Points will be colored according to distance '
                #         'from the ego vehicle instead.'.format(self.nusc.version))

        point_scale = 0.2 if sensor_modality == 'lidar' else 3.0
        scatter = ax.scatter(points[0, :], points[1, :], c=colors, s=point_scale)

        # Show velocities.
        if sensor_modality == 'radar':
            points_vel = view_points(pc.points[:3, :] + velocities, viewpoint, normalize=False)
            deltas_vel = points_vel - points
            deltas_vel = 6 * deltas_vel  # Arbitrary scaling
            max_delta = 20
            deltas_vel = np.clip(deltas_vel, -max_delta, max_delta)  # Arbitrary clipping
            colors_rgba = scatter.to_rgba(colors)
            for i in range(points.shape[1]):
                ax.arrow(points[0, i], points[1, i], deltas_vel[0, i], deltas_vel[1, i], color=colors_rgba[i])

        # Show ego vehicle.
        ax.plot(0, 0, 'x', color='red')

        # Get boxes in lidar frame.
        _, boxes, _ = nusc.get_sample_data(ref_sd_token, box_vis_level=box_vis_level,
                                                use_flat_vehicle_coordinates=use_flat_vehicle_coordinates)

        # Show boxes.
        if with_anns:
            for box in boxes:
                #c = np.array(nusc.get_color(box.name)) / 255.0
                c = np.array([0, 0, 0]) / 255.0
                box.render(ax, view=np.eye(4), colors=(c, c, c))


        if det_boxes:
            print("hello")
            sample_token = sd_record['sample_token']
            sample_rec = nusc.get('sample', sample_token)
            sd_record = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
            cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
            pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

            boxes_local = boxes_to_sensor(det_boxes, pose_record, cs_record)

            for box in boxes_local:
                #c = np.array(nusc.get_color(box.name)) / 255.0
                #if box.score > det_thresh:
                c = np.array([0, 0, 1.0]) / 255.0
                box.render(ax, view=np.eye(4), colors=('b', 'b', 'b'), linewidth=2)

    

        # Limit visible range.
        ax.set_xlim(-axes_limit, axes_limit)
        ax.set_ylim(-axes_limit, axes_limit)
    # elif sensor_modality == 'camera':
    #     # Load boxes and image.
    #     data_path, boxes, camera_intrinsic = nusc.get_sample_data(sample_data_token,
    #                                                                     box_vis_level=box_vis_level)
    #     data = Image.open(data_path)

    #     # Init axes.
    #     if ax is None:
    #         _, ax = plt.subplots(1, 1, figsize=(9, 16))

    #     # Show image.
    #     ax.imshow(data)

    #     # Show boxes.
    #     if with_anns:
    #         for box in boxes:
    #             c = np.array(self.get_color(box.name)) / 255.0
    #             box.render(ax, view=camera_intrinsic, normalize=True, colors=(c, c, c))

    #     # Limit visible range.
    #     ax.set_xlim(0, data.size[0])
    #     ax.set_ylim(data.size[1], 0)

    else:
        raise ValueError("Error: Unknown sensor modality!")

    ax.axis('off')
    ax.set_title('{} {labels_type}'.format(
        sd_record['channel'], labels_type='(predictions)' if lidarseg_preds_bin_path else ''))
    ax.set_aspect('equal')

    if out_path is not None:
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=200)

    if verbose:
        plt.show()


def render_ego_centric_map(nusc,
                            sample_data_token: str,
                            axes_limit: float = 40,
                            ax: Axes = None) -> None:
    """
    Render map centered around the associated ego pose.
    :param sample_data_token: Sample_data token.
    :param axes_limit: Axes limit measured in meters.
    :param ax: Axes onto which to render.
    """

    def crop_image(image: np.array,
                    x_px: int,
                    y_px: int,
                    axes_limit_px: int) -> np.array:
        x_min = int(x_px - axes_limit_px)
        x_max = int(x_px + axes_limit_px)
        y_min = int(y_px - axes_limit_px)
        y_max = int(y_px + axes_limit_px)

        cropped_image = image[y_min:y_max, x_min:x_max]

        return cropped_image

    # Get data.
    sd_record = nusc.get('sample_data', sample_data_token)
    sample = nusc.get('sample', sd_record['sample_token'])
    scene = nusc.get('scene', sample['scene_token'])
    log = nusc.get('log', scene['log_token'])
    map_ = nusc.get('map', log['map_token'])
    map_mask = map_['mask']
    pose = nusc.get('ego_pose', sd_record['ego_pose_token'])

    # Retrieve and crop mask.
    pixel_coords = map_mask.to_pixel_coords(pose['translation'][0], pose['translation'][1])
    scaled_limit_px = int(axes_limit * (1.0 / map_mask.resolution))
    mask_raster = map_mask.mask()
    cropped = crop_image(mask_raster, pixel_coords[0], pixel_coords[1], int(scaled_limit_px * math.sqrt(2)))

    # Rotate image.
    ypr_rad = Quaternion(pose['rotation']).yaw_pitch_roll
    yaw_deg = -math.degrees(ypr_rad[0])
    rotated_cropped = np.array(Image.fromarray(cropped).rotate(yaw_deg))

    # Crop image.
    ego_centric_map = crop_image(rotated_cropped,
                                    int(rotated_cropped.shape[1] / 2),
                                    int(rotated_cropped.shape[0] / 2),
                                    scaled_limit_px)

    # Init axes and show image.
    # Set background to white and foreground (semantic prior) to gray.
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(9, 9))
    # ego_centric_map[ego_centric_map == map_mask.foreground] = 125
    # ego_centric_map[ego_centric_map == map_mask.background] = 255
    ego_centric_map[ego_centric_map == map_mask.foreground] = 255
    ego_centric_map[ego_centric_map == map_mask.background] = 25
    ax.imshow(ego_centric_map, extent=[-axes_limit, axes_limit, -axes_limit, axes_limit],
                cmap='gray', vmin=0, vmax=255)




def render_ego_centric_map_alt(nusc,
                            sample_data_token: str,
                            axes_limit: float = 40,
                            ax: Axes = None) -> None:
    """
    Render map centered around the associated ego pose.
    :param sample_data_token: Sample_data token.
    :param axes_limit: Axes limit measured in meters.
    :param ax: Axes onto which to render.
    """

    def crop_image(image: np.array,
                    x_px: int,
                    y_px: int,
                    axes_limit_px: int) -> np.array:
        x_min = int(x_px - axes_limit_px)
        x_max = int(x_px + axes_limit_px)
        y_min = int(y_px - axes_limit_px)
        y_max = int(y_px + axes_limit_px)

        cropped_image = image[y_min:y_max, x_min:x_max]

        return cropped_image

    # Get data.
    sd_record = nusc.get('sample_data', sample_data_token)
    sample = nusc.get('sample', sd_record['sample_token'])
    scene = nusc.get('scene', sample['scene_token'])
    log = nusc.get('log', scene['log_token'])
    map_ = nusc.get('map', log['map_token'])
    map_mask = map_['mask']
    pose = nusc.get('ego_pose', sd_record['ego_pose_token'])

    # Retrieve and crop mask.
    pixel_coords = map_mask.to_pixel_coords(pose['translation'][0], pose['translation'][1])
    scaled_limit_px = int(axes_limit * (1.0 / map_mask.resolution))
    mask_raster = map_mask.mask()
    cropped = crop_image(mask_raster, pixel_coords[0], pixel_coords[1], int(scaled_limit_px * math.sqrt(2)))

    # Rotate image.
    ypr_rad = Quaternion(pose['rotation']).yaw_pitch_roll
    yaw_deg = -math.degrees(ypr_rad[0])
    rotated_cropped = np.array(Image.fromarray(cropped).rotate(yaw_deg+90))

    # Crop image.
    ego_centric_map = crop_image(rotated_cropped,
                                    int(rotated_cropped.shape[1] / 2),
                                    int(rotated_cropped.shape[0] / 2),
                                    scaled_limit_px)

    # Init axes and show image.
    # Set background to white and foreground (semantic prior) to gray.
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(9, 9))
    # ego_centric_map[ego_centric_map == map_mask.foreground] = 125
    # ego_centric_map[ego_centric_map == map_mask.background] = 255
    ego_centric_map[ego_centric_map == map_mask.foreground] = 255
    ego_centric_map[ego_centric_map == map_mask.background] = 25
    ax.imshow(ego_centric_map, extent=[-axes_limit, axes_limit, -axes_limit, axes_limit],
                cmap='gray', vmin=0, vmax=255)
