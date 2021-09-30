from typing import List
from matplotlib import pyplot as plt
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

    # Show point cloud.
    points = view_points(pc[:3, :], np.eye(4), normalize=False)
    dists = np.sqrt(np.sum(pc[:2, :] ** 2, axis=0))
    colors = np.minimum(1, dists / eval_range)
    ax.scatter(points[0, :], points[1, :], c=colors, s=0.2)

    # Show ego vehicle.
    ax.plot(0, 0, 'x', color='black')

    # Show GT boxes.
    for box in boxes_gt:
        box.render(ax, view=np.eye(4), colors=('g', 'g', 'g'), linewidth=2)

    # Show EST boxes.
    for box in boxes_est:
        # Show only predictions with a high score.
        assert not np.isnan(box.score), 'Error: Box score cannot be NaN!'
        if box.score >= conf_th:
            box.render(ax, view=np.eye(4), colors=('b', 'b', 'b'), linewidth=1)

    # Limit visible range.
    axes_limit = eval_range + 3  # Slightly bigger to include boxes that extend beyond the range.
    ax.set_xlim(-axes_limit, axes_limit)
    ax.set_ylim(-axes_limit, axes_limit)

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
        LidarPointCloud(points).render_height(ax, view=view)
       
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
        gt_box.render(ax, view=view, colors=('g', 'g', 'g'), linewidth=1)
        corners = view_points(gt_box.corners(), view, False)[:2, :]
        ax.set_xlim([np.min(corners[0, :]) - margin, np.max(corners[0, :]) + margin])
        ax.set_ylim([np.min(corners[1, :]) - margin, np.max(corners[1, :]) + margin])
        ax.axis('off')
        ax.set_aspect('equal')

        rel_pred_boxes = []
        if pred_boxes is not None:
            rel_pred_boxes = boxes_to_sensor(pred_boxes, pose_record, cs_record)
        
        for box in rel_pred_boxes:
            box.render(ax, view=view, colors=('b', 'b', 'b'), linewidth=1)


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

