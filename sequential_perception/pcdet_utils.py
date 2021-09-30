import copy
from pathlib import Path
from sequential_perception.nuscenes_utils import get_boxes_for_annotation
import numpy as np
from typing import Dict, List
from nuscenes.nuscenes import NuScenes
from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset
from pyquaternion.quaternion import Quaternion


def update_data_dict(pcdet_dataset: NuScenesDataset, old_data_dict: dict, new_points: np.array):
    # if self._merge_all_iters_to_one_epoch:
    #     index = index % len(self.infos)

    #info = copy.deepcopy(pcdet_dataset.infos[index])
    #old
    # points = pcdet_dataset.get_lidar_with_sweeps(index, max_sweeps=pcdet_dataset.dataset_cfg.MAX_SWEEPS)

    
    input_dict = {
        'points': new_points,
        'frame_id': old_data_dict['frame_id'],
        'metadata': {'token': old_data_dict['metadata']['token']}
    }

    # if 'gt_boxes' in info:
    #     if self.dataset_cfg.get('FILTER_MIN_POINTS_IN_GT', False):
    #         mask = (info['num_lidar_pts'] > self.dataset_cfg.FILTER_MIN_POINTS_IN_GT - 1)
    #     else:
    #         mask = None

    #     input_dict.update({
    #         'gt_names': info['gt_names'] if mask is None else info['gt_names'][mask],
    #         'gt_boxes': info['gt_boxes'] if mask is None else info['gt_boxes'][mask]
    #     })

    data_dict = pcdet_dataset.prepare_data(data_dict=input_dict)
    #data_dict['gt_boxes'] = old_data_dict['gt_boxes']
    # if pcdet_dataset.dataset_cfg.get('SET_NAN_VELOCITY_TO_ZEROS', False):
    #     gt_boxes = data_dict['gt_boxes']
    #     gt_boxes[np.isnan(gt_boxes)] = 0
    #     data_dict['gt_boxes'] = gt_boxes

    # if not pcdet_dataset.dataset_cfg.PRED_VELOCITY and 'gt_boxes' in data_dict:
    #     data_dict['gt_boxes'] = data_dict['gt_boxes'][:, [0, 1, 2, 3, 4, 5, 6, -1]]

    return data_dict


def get_boxes_for_pcdet_data(nusc: NuScenes, data_dict: Dict, annotation, max_sweeps=10):
    sample = nusc.get('sample', data_dict['metadata']['token'])

    ref_chan = 'LIDAR_TOP'  # The radar channel from which we track back n sweeps to aggregate the point cloud.
    chan = 'LIDAR_TOP'  # The reference channel of the current sample_rec that the point clouds are mapped to.

    ref_sd_token = sample['data'][ref_chan]
    ref_sd_rec = nusc.get('sample_data', ref_sd_token)

    sample_data_token = sample['data'][chan]
    curr_sd_rec = nusc.get('sample_data', sample_data_token)

    all_boxes = []
    while len(all_boxes) < max_sweeps - 1:
        if curr_sd_rec['prev'] == '':

            # Tranform box global->ego frame
            b = get_boxes_for_annotation(nusc, curr_sd_rec['token'], annotation['token'])
            current_pose_rec = nusc.get('ego_pose', curr_sd_rec['ego_pose_token'])
            b.translate(-np.array(current_pose_rec['translation']))
            b.rotate(Quaternion(current_pose_rec['rotation']).inverse)


            # Transform box ego->sensor frame
            current_cs_rec = nusc.get(
                'calibrated_sensor', curr_sd_rec['calibrated_sensor_token']
            )
            b.translate(-np.array(current_cs_rec['translation']))
            b.rotate(Quaternion(current_cs_rec['rotation']).inverse)

            all_boxes.append(b)
                    # break
            break
        else:
            curr_sd_rec = nusc.get('sample_data', curr_sd_rec['prev'])
                    
            # Tranform box global->ego frame
            b = get_boxes_for_annotation(nusc, curr_sd_rec['token'], annotation['token'])
            current_pose_rec = nusc.get('ego_pose', curr_sd_rec['ego_pose_token'])
            b.translate(-np.array(current_pose_rec['translation']))
            b.rotate(Quaternion(current_pose_rec['rotation']).inverse)


            # Transform box ego->sensor frame
            current_cs_rec = nusc.get(
                'calibrated_sensor', curr_sd_rec['calibrated_sensor_token']
            )
            b.translate(-np.array(current_cs_rec['translation']))
            b.rotate(Quaternion(current_cs_rec['rotation']).inverse)

            all_boxes.append(b)
        
    return all_boxes