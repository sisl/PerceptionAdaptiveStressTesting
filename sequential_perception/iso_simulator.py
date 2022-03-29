#from random import sample
import argparse
from pathlib import Path
import pickle
from random import sample
import time
from typing import Any, Dict, List
import yaml
#from unicodedata import category
import numpy as np
from nuscenes.eval.prediction.metrics import final_distances
from nuscenes.prediction.input_representation.combinators import Rasterizer
from nuscenes.prediction.input_representation.interface import InputRepresentation
from scipy.stats import bernoulli
import matplotlib.pyplot as plt

from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.prediction.config import load_prediction_config
from nuscenes.eval.prediction.data_classes import Prediction
from nuscenes.prediction.helper import PredictHelper, convert_global_coords_to_local
from nuscenes.utils.geometry_utils import points_in_box, view_points
import torch
from misc.pointpillars_adv import get_pointpillars_critical_set

from sequential_perception.classical_pipeline import  PerceptionPipeline
from sequential_perception.evaluation import compute_prediction_metrics, load_sample_gt
from sequential_perception.input_representation_tracks import AgentBoxesFromTracking, StaticLayerFromTracking
from sequential_perception.pcdet_utils import get_boxes_for_pcdet_data, update_data_dict
from sequential_perception.nuscenes_utils import get_ordered_samples, render_box_with_pc, vis_sample_pointcloud
from sequential_perception.disturbances import BetaRadomization, haze_point_cloud
from sequential_perception.predict_helper_tracks import TrackingResultsPredictHelper
from sequential_perception.utils import build_pipeline, build_predictor, build_tracker, build_detector


from typing import Callable

import numpy as np

from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.utils import center_distance, scale_iou, yaw_diff, velocity_l2, attr_acc, cummean
from nuscenes.eval.detection.data_classes import DetectionMetricData

from nuscenes.utils.geometry_utils import points_in_box
from nuscenes.utils.data_classes import Box

from pyquaternion import Quaternion
from pcdet.models import build_network, load_data_to_gpu


def accumulate(gt_boxes: EvalBoxes,
               pred_boxes: EvalBoxes,
               class_name: str,
               dist_fcn: Callable,
               dist_th: float,
               verbose: bool = False):

    # Organize the predictions in a single list.
    pred_boxes_list = [box for box in pred_boxes.all if box.detection_name == class_name]
    pred_confs = [box.detection_score for box in pred_boxes_list]

    if verbose:
        print("Found {} PRED of class {} out of {} total across {} samples.".
              format(len(pred_confs), class_name, len(pred_boxes.all), len(pred_boxes.sample_tokens)))

    # Sort by confidence.
    sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(pred_confs))][::-1]

    # Do the actual matching.
    tp = []  # Accumulator of true positives.
    fp = []  # Accumulator of false positives.
    conf = []  # Accumulator of confidences.

    # match_data holds the extra metrics we calculate for each match.
    match_data = {'trans_err': [],
                  'vel_err': [],
                  'scale_err': [],
                  'orient_err': [],
                  'attr_err': [],
                  'conf': []}

    # ---------------------------------------------
    # Match and accumulate match data.
    # ---------------------------------------------
    taken = set()  # Initially no gt bounding box is matched.
    matched_pred_boxes = []
    for ind in sortind:
        pred_box = pred_boxes_list[ind]
        min_dist = np.inf
        match_gt_idx = None

        for gt_idx, gt_box in enumerate(gt_boxes[pred_box.sample_token]):

            # Find closest match among ground truth boxes
            if gt_box.detection_name == class_name and not (pred_box.sample_token, gt_idx) in taken:
                this_distance = dist_fcn(gt_box, pred_box)
                if this_distance < min_dist:
                    min_dist = this_distance
                    match_gt_idx = gt_idx

        # If the closest match is close enough according to threshold we have a match!
        is_match = min_dist < dist_th
        if is_match:
            taken.add((pred_box.sample_token, match_gt_idx))
            matched_pred_boxes.append(pred_box)

    taken_gt_idxs = [e[1] for e in taken]
    gt_boxes_list = gt_boxes[pred_box.sample_token]
    matched_gt_boxes = [gt_boxes_list[i] for i in taken_gt_idxs]

    return matched_gt_boxes, matched_pred_boxes

def free_data_from_gpu(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, torch.Tensor):
            continue
        elif key in ['frame_id', 'metadata', 'calib']:
            continue
        elif key in ['image_shape']:
            batch_dict[key] = val.cpu()
        else:
            batch_dict[key] = val.cpu()


def get_critical_set(pcdet_module, data_dict, lidar_frame_boxes):
    batch_dict = pcdet_module.pcdet_dataset.collate_batch([data_dict])
    load_data_to_gpu(batch_dict)
    with torch.no_grad():
        critical_sets = get_pointpillars_critical_set(pcdet_module.model, batch_dict)
    critical_set_array = np.vstack(list(critical_sets.values()))
    critical_set_array = critical_set_array[~np.all(critical_set_array == 0, axis=1)]
    critical_set_array  = np.unique(critical_set_array , axis=0)


    masks = [points_in_box(b, critical_set_array[:, :3].T, wlh_factor=1) for b in lidar_frame_boxes]
    overall_mask = np.logical_or.reduce(np.array(masks))
    idxs_in_boxes = overall_mask.nonzero()[0]

    reduced_critical_set = critical_set_array[idxs_in_boxes, :]
    #free_data_from_gpu(batch_dict
    del batch_dict


    return reduced_critical_set
def fast_iso_pointpillars(data_dict, gt_boxes, model, predict, monotone=True):
    "Adapted from https://github.com/matthewwicker/IterativeSalienceOcclusion"

    confidences = [1]
    removed = []
    removed_ind = []
    points_occluded = 0
    points = data_dict['points']

    x = list(points)
    rho = np.linalg.norm(points[:, :3], axis=1)

    # Find boxes for sample in sensor frame 
    nusc = model.nuscenes
    sample_token = data_dict['metadata']['token']
    sample_lidar_token = nusc.get('sample', sample_token)['data']['LIDAR_TOP']

    gt_boxtype = [Box(b.translation, b.size, Quaternion(b.rotation)) for b in gt_boxes]
    sd_record = nusc.get('sample_data', sample_lidar_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
    
    lidar_frame_boxes = []
    for box in gt_boxtype:
        # Move box to ego vehicle coord system.
        box.translate(-np.array(pose_record['translation']))
        box.rotate(Quaternion(pose_record['rotation']).inverse)

        #  Move box to sensor coord system.
        box.translate(-np.array(cs_record['translation']))
        box.rotate(Quaternion(cs_record['rotation']).inverse)

        lidar_frame_boxes.append(box)
  
    # conf_i, cl = predict(x, model)
    conf_i, cl = predict(x, data_dict, gt_boxes, model)
    
    # Calculate the critical set
    iterations = 0
    new_points = np.asarray(x)
    all_cs = []
    while(True):
        iterations += 1
        #cs = get_cs(model, x)
        xyz_zero_mask = np.sum(new_points[:, :3], axis=1) != 0.0
        new_points = new_points[xyz_zero_mask, :]
        new_data_dict = update_data_dict(model.pcdet_dataset, data_dict, new_points)
        #cs = get_cs(model.model, new_data_dict)
        cs = get_critical_set(model, new_data_dict, lidar_frame_boxes)
        card_cs = len(cs)
        # Invert sort the critical set by the distance to the nearest neighbor
        cs_range = cs[:, 2]

        cs_ranking = np.argsort(cs_range)
        cs = cs[cs_ranking, :]
        cs_len = np.min([cs.shape[0], 5])
        cs = cs[:cs_len, :]
        all_cs.append(cs)

        cs_idxs = np.argwhere(np.isin(new_points, cs).all(axis=1)).flatten()
        all_idxs = np.arange(0, new_points.shape[0])

        new_points_idxs = np.setdiff1d(all_idxs, cs_idxs)

        new_points = new_points[new_points_idxs, :]

        conf, cl = predict(new_points, data_dict, gt_boxes, model)
        if(cl != 1):
            print('Yay')
            print(iterations)
            return new_points, np.vstack(all_cs)

        if iterations >= 100 or cs.shape[0] == 0:
            print(iterations)
            return new_points, np.vstack(all_cs)


def iso_predict(points, data_dict, true_boxes, model):
    points = np.asarray(points)
    xyz_zero_mask = np.sum(points[:, :3], axis=1) != 0.0
    points = points[xyz_zero_mask, :]
    #print("Predict #Removed: {}".format(np.sum(np.logical_not(xyz_zero_mask))))
    new_data_dict = update_data_dict(model.pcdet_dataset, data_dict, points)
    dets = model([new_data_dict])

    det_boxes = EvalBoxes.deserialize(dets['results'], DetectionBox)
    gt_evalboxes = EvalBoxes()
    gt_evalboxes.add_boxes(true_boxes[0].sample_token, true_boxes)

    gt_matched, det_matched = accumulate(gt_evalboxes, det_boxes, "car", center_distance, dist_th = 2.0)
    if len(true_boxes) != len(gt_matched):
        conf = 0
        cl = 0
    else:
        cl = 1
        conf = np.max([d.detection_score for d in det_matched])

    return conf, cl


class ISOPerceptionSimulator:
    def __init__(self,
                 nuscenes: NuScenes,
                 pipeline: PerceptionPipeline,
                 scene_infos_list,
                 pipeline_config: Dict = {},
                 scene_token: str = 'fcbccedd61424f1b85dcbf8f897f9754',
                 warmup_steps=4,
                 scatter_fraction = 0.05, # [-]
                 max_range = 25,
                 eval_metric = 'MinFDEK',
                 eval_k = 10,
                 eval_metric_th=10.0,
                 no_pred_failure=True,
                 logger=None) -> None:

        assert eval_metric in ['MinADEK', 'MinFDEK']
        assert eval_k in [1, 5, 10]

        self.nuscenes = nuscenes
        self.pipeline = pipeline
        self.pipeline_config = pipeline_config
        self.pcdet_dataset = pipeline.detector.pcdet_dataset
        self.no_pred_failure = no_pred_failure # Fail if no prediction during an eval sample?
        self._action = None
        self.predict_eval_config_name = 'predict_2020_icra.json'
        nusc_helper = PredictHelper(self.nuscenes)
        self.predict_eval_config = load_prediction_config(nusc_helper, self.predict_eval_config_name)
        #scene_token = nuscenes.scene[0]
        self.max_range = max_range

        # Disturbance params
        self.scatter_fraction = scatter_fraction

        self.s0 = 0
        self.eval_idx = self.s0
        #self.horizon = len(self.eval_samples)
        self.step = 0
        self.action = None
        self.action_mag = 0
        self.info = []
        self.best_reward = -np.inf
        self.points_log = {}

        self.eval_metric = eval_metric
        self.eval_k = eval_k
        self.eval_metric_th = eval_metric_th
    
        self.sim_log = {'actions': [],
                    'reward': 0.0,
                    'failure_sample': '',
                    'failure_agents': [],
                    'failure_metrics': [],
                    }
        self.failure_log = []
        self.failure_perception_data = []
        
        # Get all samples for scene
        ordered_samples = get_ordered_samples(nuscenes, scene_token)
        #ordered_samples = ordered_samples[1:]
        assert len(ordered_samples) > 30
        # Iterate over each instance in the scene

        # Load all PCDet data for scene
        pcdet_dataset = pipeline.detector.pcdet_dataset
        self.pcdet_infos = {}
        for i in range(len(scene_infos_list)):
            data_dict = scene_infos_list[i]
            if data_dict['metadata']['token'] in ordered_samples:
                self.pcdet_infos[data_dict['metadata']['token']] = data_dict

        # map tokens -> future trajectories
        pred_candidate_info = {}

        for inst_record in self.nuscenes.instance:
            cat = self.nuscenes.get('category', inst_record['category_token'])
            
            # If this is a vehicle we can predict for
            if np.any([v in cat['name'] for v in ['car', 'bus', 'truck']]):

                # Iterate through each annotation of this instance
                ann_tokens = []
                first_ann_token = inst_record['first_annotation_token']
                cur_ann_record = self.nuscenes.get('sample_annotation', first_ann_token)
                while cur_ann_record['next'] != '':
                    ann_tokens.append(cur_ann_record['token'])
                    cur_ann_record = self.nuscenes.get('sample_annotation', cur_ann_record['next'])

                # Must have at least 6 seconds (12 annotations) of in scene
                if len(ann_tokens) < 12:
                    continue
                
                consecutive_tsteps_in_range = 0
                min_tsteps = 3
                
                for ann_token in ann_tokens:
                    ann_record = self.nuscenes.get('sample_annotation', ann_token)

                    if ann_record['sample_token'] == ordered_samples[0]:
                        continue

                    current_attr = self.nuscenes.get('attribute', ann_record['attribute_tokens'][0])['name']
                    if 'stopped' in current_attr:
                        continue

                    # get range of annotation
                    sample_token = ann_record['sample_token']
                    sample_rec = self.nuscenes.get('sample', sample_token)
                    sd_record = self.nuscenes.get('sample_data', sample_rec['data']['LIDAR_TOP'])
                    pose_record = self.nuscenes.get('ego_pose', sd_record['ego_pose_token'])

                    # Both boxes and ego pose are given in global coord system, so distance can be calculated directly.
                    box = self.nuscenes.get_box(ann_record['token'])
                    ego_translation = (box.center[0] - pose_record['translation'][0],
                                    box.center[1] - pose_record['translation'][1],
                                    box.center[2] - pose_record['translation'][2])

                    ann_range = np.linalg.norm(ego_translation)

                    if ann_range <= self.max_range:
                        consecutive_tsteps_in_range += 1
                    
                    if consecutive_tsteps_in_range >= min_tsteps:

                        # Can we get the next six seconds?
                        try:
                            ann_future = nusc_helper.get_future_for_agent(inst_record['token'],
                                                            ann_record['sample_token'],
                                                            seconds=6.0,
                                                            in_agent_frame=False)
                            if ann_future.shape[0] >= 12:
                                pred_token = inst_record['token'] + '_' + ann_record['sample_token']

                                pred_candidate_info[pred_token] = ann_future

                        except:
                            pass
        
        scene_data_dicts = list(self.pcdet_infos.values())
        all_pred_tokens = list(pred_candidate_info.keys())
        dets, tracks, preds = self.pipeline(scene_data_dicts, all_pred_tokens, reset=True)

        # calculate good detections at each time step
        self.gt_matches_map = {}
        nominal_boxes = EvalBoxes.deserialize(dets['results'], DetectionBox)
        for t in ordered_samples:
            t_dets = nominal_boxes[t]
            # Get GT box matches in nominal case
            gt_boxes = load_sample_gt(self.nuscenes, t, DetectionBox)
            t_eval_boxes = EvalBoxes()
            t_eval_boxes.add_boxes(t, t_dets)
            nominal_gt_matches, _ = accumulate(gt_boxes, t_eval_boxes, "car", center_distance, dist_th = 2.0)
            self.gt_matches_map[t] = nominal_gt_matches

        self.init_preds = preds
        self.pred_candidate_info = pred_candidate_info

        prediction_list = [Prediction.deserialize(p) for p in self.init_preds]
        passing_pred_tokens = []
        for pred in prediction_list:
            pred_token = pred.instance + '_' + pred.sample
            gt_pred = self.pred_candidate_info[pred_token]
            sample_metrics = compute_prediction_metrics(pred, gt_pred, self.predict_eval_config)

            eval_idx = self.eval_k // 5
            eval_value = sample_metrics[self.eval_metric][pred.instance][0][eval_idx]
            if eval_value < self.eval_metric_th:
                passing_pred_tokens.append(pred_token)

        self.passing_pred_tokens = passing_pred_tokens

        # get unique samples where we want to run evaluation (find latest sample)
        token_sample_idxs = [ordered_samples.index(t.split('_')[1]) for t in self.passing_pred_tokens]
        self.last_sample_idx = np.max(token_sample_idxs)
        self.first_sample_idx = np.min(token_sample_idxs) - 3
        self.first_sample_idx = np.max([self.first_sample_idx, 1])

        self.horizon = (self.last_sample_idx - self.first_sample_idx) + 1
        
        self.ordered_sample_tokens = ordered_samples

        # Map sample tokens to lists of pred tokens
        sample2predtokens = {}
        for sample_token in self.ordered_sample_tokens:
            pred_tokens = [t for t in self.passing_pred_tokens if sample_token in t]
            sample2predtokens[sample_token] = pred_tokens

        self.sample2predtokens = sample2predtokens

        self.ordered_sample_tokens = self.ordered_sample_tokens[self.first_sample_idx-1:self.last_sample_idx]

        self.reset(self.s0)


    def simulate(self, render=False):
        path_length = 0
        self.reset(0)
        self.info = []
        simulation_horizon = self.horizon

        while path_length < simulation_horizon:
            self.step_simulation()

            if self.is_goal():
                return path_length, np.array(self.info)

            path_length += 1

        self.terminal = True

        return -1, np.array(self.info)

    def step_simulation(self):
        idx = self.step

        sample_token = self.ordered_sample_tokens[idx] #  sample_token = self.eval_samples[idx]

        data_dict = self.pcdet_infos[sample_token]
        points = data_dict['points']
        gt_boxes = self.gt_matches_map[sample_token]
        model = self.pipeline.detector
        if len(gt_boxes) > 0:
            new_points, cs = fast_iso_pointpillars(data_dict, gt_boxes, model, iso_predict)
        else:
            new_points = points
            cs = np.array([])
        self.new_points = new_points
        self.points_log[sample_token] = new_points
        self.cs_log[sample_token] = cs
        self.action_mag = data_dict['points'].shape[0] - self.new_points.shape[0]
        self.action_frac = self.action_mag/data_dict['points'].shape[0]
        new_data_dict = update_data_dict(self.pipeline.detector.pcdet_dataset, data_dict, new_points[:, :5])

        detections = self.pipeline.run_detection([new_data_dict])
        tracks = self.pipeline.run_tracking(detections['results'], batch_mode=False)
        pred_tokens = [t for t in self.passing_pred_tokens if sample_token in t]
        if len(pred_tokens) > 0:
            predictions = self.pipeline.run_prediction(tracks, pred_tokens)
            prediction_list = [Prediction.deserialize(p) for p in predictions]
        else:
            prediction_list = []

        # Log
        det_boxes = EvalBoxes.deserialize(detections['results'], DetectionBox)
        self.detections.append(det_boxes)
        self.tracks.append(tracks)
        self.predictions.append(prediction_list)
        self.step += 1
        self.log()
        return self.s0

    def reset(self, s0):
        # Reset pipeline
        self.step = s0
        self.action = None
        self.detections = []
        self.tracks = []
        self.predictions = []
        self.pipeline.reset()
        self.cnt = 0
        self.action_prob = 1.0
        self.log_action_prob = 0.0
        self.action_mag = 0
        self.points_log = {}
        self.cs_log = {}

        self.sim_log = {'actions': [],
                'action_magnitudes': [],
                'action_fraction': [],
                'detections':[],
                'tracks':[],
                'predictions':[],
                'reward': 0.0,
                'failure_sample': '',
                'failure_agents': [],
                'failure_metrics': [],
                }

        return s0

    def is_goal(self):
        self.failure_instances = []
        self.failure_eval_metrics = []
        self.failure_sample = None
        if self.step == 0:
            return False 

        sample_token = self.ordered_sample_tokens[self.step-1]
        num_preds = len([p for p in self.predictions[-1] if p])
        if num_preds != len(self.sample2predtokens[sample_token]) and self.no_pred_failure:
            current_pred_tokens = [pred.instance + '_' + pred.sample for pred in self.predictions[-1]]
            for gt_pred_token in self.sample2predtokens[sample_token]:
                if gt_pred_token not in current_pred_tokens:
                    self.failure_eval_metrics.append('miss')
                    self.failure_instances.append(gt_pred_token.split('_')[0])
                    self.failure_sample = gt_pred_token.split('_')[1]
            return True
        
        # If there is a prediction, check if avg error greater than threshold
        goal = False

        for pred in self.predictions[-1]:
            pred_token = pred.instance + '_' + pred.sample
            gt_pred = self.pred_candidate_info[pred_token]
            sample_metrics = compute_prediction_metrics(pred, gt_pred, self.predict_eval_config)

            eval_idx = self.eval_k // 5
            eval_value = sample_metrics[self.eval_metric][pred.instance][0][eval_idx]
            if eval_value > self.eval_metric_th:
                goal = True
        
        return goal

    def is_terminal(self):
        return self.step >= self.horizon

    def log(self):
        self.sim_log['action_magnitudes'].append(self.action_mag)
        self.sim_log['action_fraction'].append(self.action_frac)
        self.sim_log['reward'] += self.log_action_prob
        if self.is_goal():
            self.sim_log['failure_sample'] = self.failure_sample
            self.sim_log['failure_agents'] = self.failure_instances
            self.sim_log['failure_metrics'] = self.failure_eval_metrics

            ann_tokens = self.nuscenes.field2token('sample_annotation', 'instance_token', self.failure_instances[0])
            dist_points_in_instance = []
            nom_points_in_instance = []
            for i in range(len(self.ordered_sample_tokens)):
                tok = self.ordered_sample_tokens[i]
                if tok in list(self.points_log.keys()):
                    sample_data_tok = self.nuscenes.get('sample', tok)['data']['LIDAR_TOP']
                    _, box_list, _ = self.nuscenes.get_sample_data(sample_data_tok)
                    for box in box_list:
                        if box.token in ann_tokens:
                            dist_points_in_box = points_in_box(box, self.points_log[tok][:, :3].T)
                            dist_points_in_instance.append(np.sum(dist_points_in_box))
                            nom_points_in_box = points_in_box(box, self.pcdet_infos[tok]['points'][:, :3].T)
                            nom_points_in_instance.append(np.sum(nom_points_in_box))

            self.sim_log['dist_points_in_instance'] = dist_points_in_instance
            self.sim_log['nom_points_in_instance'] = nom_points_in_instance
            self.best_reward = self.sim_log['reward']
            self.failure_log.append([self.sim_log])
            self.failure_perception_data = {'detections': self.detections,
                                            'tracks': self.tracks,
                                            'predictions':self.predictions
                                            }

    def observation(self):
        return self.s0

    def get_action_prob(self, ptrue, action):
        return np.prod([p**action[i] * (1-p)**action[i] for i,p in enumerate(ptrue)])

    def get_predict_tokens(self):
        predict_tokens = []
        for sample_token in self.eval_samples:
            predict_tokens.append(self.target_instance + '_' + sample_token)
        return predict_tokens

    def get_pointcloud(self, sample_token, action):
        points = self.pcdet_infos[sample_token]['points']

        idxs_in_box = self.sample_to_point_idxs[sample_token]
        idxs_remove_mask = bernoulli.rvs(self.drop_likelihood,
                                            size=idxs_in_box.shape[0],
                                            random_state=action)
        idxs_to_remove = idxs_in_box[idxs_remove_mask>0]
        new_points = np.delete(points, idxs_to_remove, axis=0)
        return new_points


    def _render_detection(self, render_path):
        # If no detections, do nothing
        if len(self.detections) == 0:
            return

        det_boxes = self.detections[-1]
        sample_token = det_boxes.sample_tokens[0]
        sample_rec = self.nuscenes.get('sample', sample_token)
        sd_record = self.nuscenes.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        cs_record = self.nuscenes.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        pose_record = self.nuscenes.get('ego_pose', sd_record['ego_pose_token'])
        gt_boxes = load_sample_gt(self.nuscenes, sample_token, DetectionBox)
        points = self.points_log[sample_token]
        cs = self.cs_log[sample_token]

        ax = vis_sample_pointcloud(self.nuscenes,
                                   sample_token,
                                   gt_boxes=gt_boxes,
                                   pred_boxes=det_boxes,
                                   #pred_boxes=[],
                                   # pc=self.pcdet_infos[sample]['points'][self.sample_to_point_idxs[sample], :].T,
                                   pc=points.T,
                                   savepath=None)
        plt.sca(ax)

        ax.scatter(cs[:, 0], cs[:, 1], c='r', s=0.2)
        savepath = render_path + str(self.step) + '_'+ sample_token
        plt.savefig(savepath, bbox_inches='tight', dpi=300)

        return ax

    def _render_prediction(self, render_path):
        import matplotlib.cm
        from nuscenes.eval.prediction.metrics import stack_ground_truth, mean_distances

        if len(self.predictions[-1]) == 0:
            return

        track_helper = TrackingResultsPredictHelper(self.nuscenes, self.tracks[-1]['results'])
        agent_rasterizer = AgentBoxesFromTracking(track_helper, seconds_of_history=self.pipeline.predictor.seconds_of_history)
        map_rasterizer = StaticLayerFromTracking(track_helper)
        input_representation = InputRepresentation(map_rasterizer, agent_rasterizer, Rasterizer())

        for i in range(len(self.predictions[-1])):
            prediction = self.predictions[-1][i]
            sample_token = prediction.sample
            inst_token = prediction.instance

            output_probs = prediction.probabilities

            output_trajs_global = prediction.prediction

            nearest_track = self.pipeline.predictor._find_nearest_track(track_helper, inst_token, sample_token)
            if nearest_track == None:
                continue
            track_instance = nearest_track['tracking_id']

            input_image = input_representation.make_input_representation(track_instance, sample_token)
    
            # Convert output trajs global --> local
            center_agent_annotation = track_helper.get_sample_annotation(track_instance, sample_token)
            center_translation = center_agent_annotation['translation'][:2]
            center_rotation = center_agent_annotation['rotation']
            output_trajs = np.zeros(output_trajs_global.shape)
            for i in range(output_trajs_global.shape[0]):
                global_traj = output_trajs_global[i, :, :]
                output_trajs[i, :, :] = convert_global_coords_to_local(global_traj, center_translation, center_rotation)

            pred_token = prediction.instance + '_' + prediction.sample
            gt_future = self.pred_candidate_info[pred_token]
            predicted_track = nearest_track
            gt_future_local = convert_global_coords_to_local(gt_future, predicted_track['translation'], predicted_track['rotation'])

            stacked_ground_truth = stack_ground_truth(gt_future_local, prediction.number_of_modes)
            mean_dists = mean_distances(np.array(output_trajs), stacked_ground_truth).flatten()
            final_dists = final_distances(np.array(output_trajs), stacked_ground_truth).flatten()
            dist_sorted_idxs = np.argsort(mean_dists)

            savepath = render_path + str(self.step) + '_' + inst_token + '_' + sample_token
            cmap = matplotlib.cm.get_cmap('tab10')
            msize=7
            lwidth = 1.4
            fig, ax = plt.subplots(1, 1, figsize=(9, 9))
            plt.imshow(input_image)


            if 'ADE' in self.eval_metric:
                eval_arr = mean_dists
            else:
                eval_arr = final_dists
            for i in range(5):
                traj = output_trajs[i]
                label = 'p={:4.3f}, {}{}={:4.2f}'.format(output_probs[i], self.eval_metric[3:-1], self.eval_k, eval_arr[i])
                plt.plot(10.*traj[:, 0] + 250, -(10.*traj[:, 1]) + 400, color=cmap(i), marker='o', markersize=msize, label=label, zorder=10-i)

            plt.plot(10.*gt_future_local[:, 0] + 250, -(10.*gt_future_local[:, 1]) + 400, color='black', marker='o', markersize=msize, linewidth=lwidth, label='GT')
            leg = plt.legend(frameon=True, facecolor='white', framealpha=1, prop={'size': 14})
            plt.xlim(0, 500)
            plt.ylim(500, 0)
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
                    
            plt.savefig(savepath, dpi=600)
        plt.close('all')

