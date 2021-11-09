from random import sample
import time
from typing import Any, Dict, List
from unicodedata import category
import numpy as np
from numpy.core.fromnumeric import cumsum
from scipy.stats import bernoulli
import matplotlib.pyplot as plt

from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.prediction.config import load_prediction_config
from nuscenes.eval.prediction.data_classes import Prediction
from nuscenes.prediction.helper import PredictHelper, convert_global_coords_to_local
from nuscenes.utils.geometry_utils import points_in_box, view_points

from sequential_perception.classical_pipeline import  PerceptionPipeline
from sequential_perception.evaluation import compute_prediction_metrics, load_sample_gt
from sequential_perception.pcdet_utils import get_boxes_for_pcdet_data, update_data_dict
from sequential_perception.nuscenes_utils import get_ordered_samples, render_box_with_pc, vis_sample_pointcloud
from sequential_perception.disturbances import BetaRadomization, haze_point_cloud

class FogScenePerceptionSimulator:
    def __init__(self,
                 nuscenes: NuScenes,
                 pipeline: PerceptionPipeline,
                 pipeline_config: Dict = {},
                 scene_token: str = 'fcbccedd61424f1b85dcbf8f897f9754',
                 # target_instance: str = '045cd82a77a1472499e8c15100cb5ff3',
                 #eval_samples: List = [],
                 warmup_steps=4,
                 scatter_fraction = 0.05, # [-]
                 fog_density = 0.005, # 0.0 - 0.08 [1/m]
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

        self.max_range = max_range

        # Disturbance params
        self.beta = BetaRadomization(fog_density, 0)
        self.scatter_fraction = scatter_fraction
        self.fog_density = fog_density

        self.s0 = 0
        self.eval_idx = self.s0
        #self.horizon = len(self.eval_samples)
        self.step = 0
        self.action = None
        self.info = []

        self.eval_metric = eval_metric
        self.eval_k = eval_k
        self.eval_metric_th = eval_metric_th
        
        self.detections = []
        self.tracks = []
        self.predictions = []
        

        # Get all samples for scene
        ordered_samples = get_ordered_samples(nuscenes, scene_token)
        #ordered_samples = ordered_samples[1:]
        assert len(ordered_samples) > 30

        #self.eval_samples = []

        # build a list of predict_tokens from every sample in the scene based on:
        # 1. The vehicle is not parked
        # 2. The vehicle is within max_range meters of ego

        
        # for sample_token in ordered_samples:
        #     # get all non parked cars, buses, trucks
        #     anns = nusc_helper.get_annotations_for_sample(sample_token)

        #     # For each annotation, create pred token if
        #     # - Annotation is a vehicle (car, truck, bus category)
        #     # - Vehicle is within max_range
        #     # - Vehicle has at least seconds_of_prediction time left in the scene
        #     for ann in anns:
                
        #         if 'car' in ann['category_name'] or 'bus' in ann['category_name'] or 'truck' in ann['category_name']:
        #             instance_token = ann['instance_token']



        # Iterate over each instance in the scene

        # map tokens -> future trajectories
        pred_candidate_info = {}

        for inst_record in self.nuscenes.instance:
            cat = self.nuscenes.get('category', inst_record['category_token'])
            
            # If this is a vehicle we can predict for
            if np.any([v in cat['name'] for v in ['car', 'bus', 'truck']]):

                # Iterate through each annotation of this instance

                #ann_tokens = self.nuscenes.field2token('sample_annotation', 'instance_token', inst_record['token'])
                ann_tokens = []
                first_ann_token = inst_record['first_annotation_token']
                cur_ann_record = self.nuscenes.get('sample_annotation', first_ann_token)
                while cur_ann_record['next'] != '':
                    ann_tokens.append(cur_ann_record['token'])
                    cur_ann_record = self.nuscenes.get('sample_annotation', cur_ann_record['next'])


                # Must have at least 6 seconds (12 annotations) of in scene
                if len(ann_tokens) < 12:
                    continue

                # Is it driving (not parked)?
                # Has it been in range for at least n steps?
                # Are there at least 6 consecutive seconds remaining in the scene?
                # If yes to all above - make a prediction
                
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

                            #print('Good annotation!')
                            #print(ann_record)
                            # generate a prediction token
                            if ann_future.shape[0] >= 12:
                                pred_token = inst_record['token'] + '_' + ann_record['sample_token']

                                pred_candidate_info[pred_token] = ann_future

                        except:
                            pass
        

        scene_data_dicts = [self.pcdet_dataset[i] for i in range(len(self.pcdet_dataset))]
        all_pred_tokens = list(pred_candidate_info.keys())
        dets, tracks, preds = self.pipeline(scene_data_dicts, all_pred_tokens, reset=True)
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
            # minadek = sample_metrics['MinADEK'][self.target_instance][0][-1]
            # minade5 = sample_metrics['MinADEK'][self.target_instance][0][-2]
            # minfde = sample_metrics['MinFDEK'][self.target_instance][0][-1]
            # missrate = sample_metrics['MissRateTopK_2'][self.target_instance]

            print('{} value: {}'.format(self.eval_metric, eval_value))
            if eval_value < self.eval_metric_th:
                passing_pred_tokens.append(pred_token)

        self.passing_pred_tokens = passing_pred_tokens


        # Run pipeline on true data
        #scene_data_dicts = [self.pcdet_dataset[i] for i in range(len(self.pcdet_dataset))]
        #scene_data_dicts = scene_data_dicts[1:]
        #dets, tracks, preds = self.pipeline(scene_data_dicts, reset=True)





        # find samples where instance is active
        # target_ann_tokens = set(nuscenes.field2token('sample_annotation', 'instance_token', target_instance))
        # target_annotations = [nuscenes.get('sample_annotation', t) for t in target_ann_tokens]
        # target_annotations = sorted(target_annotations, key=lambda x: ordered_samples.index(x['sample_token']))
        
        # last_target_sample_token = target_annotations[-1]['sample_token']
        # ordered_samples = ordered_samples[:ordered_samples.index(last_target_sample_token)]
        # self.ordered_samples = ordered_samples[:len(ordered_samples)-12]
        
        # target_annotations = [ann for ann in target_annotations if ann['sample_token'] in self.ordered_samples]
    
        # # filter samples for when target instance is within in ~20 m
        # filtered_target_annotations = []
        # for idx, sample_ann in enumerate(target_annotations):
        #     sample_token = sample_ann['sample_token']
        #     sample_rec = nuscenes.get('sample', sample_token)
        #     sd_record = nuscenes.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        #     pose_record = nuscenes.get('ego_pose', sd_record['ego_pose_token'])


        #     # Both boxes and ego pose are given in global coord system, so distance can be calculated directly.
        #     box = nuscenes.get_box(sample_ann['token'])
        #     ego_translation = (box.center[0] - pose_record['translation'][0],
        #                        box.center[1] - pose_record['translation'][1],
        #                        box.center[2] - pose_record['translation'][2])
        #     if np.linalg.norm(ego_translation) <= max_range:
        #         filtered_target_annotations.append(sample_ann)

        # self.target_annotations = filtered_target_annotations
        # self.target_samples = [ann['sample_token'] for ann in self.target_annotations]

        # assert len(self.target_samples) > 2, \
        #        'At least 2 valid samples required, found {}'.format(len(self.target_samples))

        # if len(eval_samples) == 0:
        #     self.eval_samples = self.target_samples
        #     self.eval_annotations = self.target_annotations
        # else:
        #     self.eval_samples = eval_samples
        #     self.eval_annotations = [ann for ann in self.target_annotations if ann['sample_token'] in self.eval_samples]


        # [print(self.ordered_samples.index(t)) for t in self.eval_samples]

        # # Load all PCDet data for scene
        # pcdet_dataset = pipeline.pcdet_dataset
        # self.pcdet_infos = {}
        # for data_dict in pcdet_dataset:
        #     if data_dict['metadata']['token'] in self.ordered_samples:
        #         self.pcdet_infos[data_dict['metadata']['token']] = data_dict

        # # Get boxes for target instance
        # #sample_to_target_boxes = {}
        # self.sample_to_point_idxs = {}
        # self.sample_to_boxes = {}
        # self.sample_to_gt_pred = {}
        # nusc_helper = PredictHelper(self.nuscenes)
        # self.predict_eval_config = load_prediction_config(nusc_helper, self.predict_eval_config_name)
        # for ann in self.eval_annotations:
        #     data_dict = self.pcdet_infos[ann['sample_token']]
        #     boxes = get_boxes_for_pcdet_data(self.nuscenes, data_dict, ann)
        #     self.sample_to_boxes[ann['sample_token']] = boxes
        
        #     points = data_dict['points'] # [n x 5]

        #     # masks = [points_in_box(b, points[:, :3].T, wlh_factor=np.array([1.2, 1.2, 10])) for b in boxes]
        #     masks = [points_in_box(b, points[:, :3].T, wlh_factor=1) for b in boxes]
        #     overall_mask = np.logical_or.reduce(np.array(masks))
        #     idxs_in_box = overall_mask.nonzero()[0]

        #     self.sample_to_point_idxs[ann['sample_token']] = idxs_in_box


        #     # Run detection for samples outside eval_samples 
        #     base_data_dicts = [d for key,d in self.pcdet_infos.items() if key not in self.eval_samples]
        #     self.base_detections = self.pipeline.run_detection(base_data_dicts)

        #     # Pre-calculate ground truth trajectory predictions
        #     sample_future = nusc_helper.get_future_for_agent(self.target_instance,
        #                                                      ann['sample_token'],
        #                                                      seconds=6.0,
        #                                                      in_agent_frame=False)
        #     self.sample_to_gt_pred[ann['sample_token']] = sample_future

        # # Get predict tokens
        # self.predict_tokens = self.get_predict_tokens()

        self.reset(self.s0)


    def simulate(self, actions: List[int], s0: int, render=False):
        path_length = 0
        self.reset(s0)
        self.info = []
        simulation_horizon = np.minimum(self.horizon, len(actions))

        #print('--SIM START--')

        # data_dicts = {key:self.pcdet_infos[key] for key in self.pcdet_infos if key in self.eval_samples}

        loop_start = time.time()
        #for idx, action in enumerate(actions):
        while path_length < simulation_horizon:
            self.action = actions[path_length]

            self.step_simulation(self.action)

            if self.is_goal():
                return path_length, np.array(self.info)

            path_length += 1

        self.terminal = True
        
        loop_end = time.time()
        # print('Loop time: {}'.format(loop_end - loop_start))

        return -1, np.array(self.info)

    def step_simulation(self, action: int):
        idx = self.step
        self.action = action
        sample_token = self.eval_samples[idx]
        print('STEP: {}'.format(self.step))

        data_dict = self.pcdet_infos[sample_token]
        points = data_dict['points']
        # idxs_in_box = self.sample_to_point_idxs[sample_token]

        # # Select point to remove
        # idxs_remove_mask = bernoulli.rvs(self.drop_likelihood,
        #                                     size=idxs_in_box.shape[0],
        #                                     random_state=action)
        # self.action_prob = self.get_action_prob(idxs_remove_mask)
        # idxs_to_remove = idxs_in_box[idxs_remove_mask>0]

        # # Remove points and update detection input
        # new_points = np.delete(points, idxs_to_remove, axis=0)
        new_points, action_prob = haze_point_cloud(points, self.beta, self.scatter_fraction, action)
        #new_points = new_points[:, :5]
        self.new_points = new_points
        new_data_dict = update_data_dict(self.pipeline.pcdet_dataset, data_dict, new_points[:, :5])
        self.action_prob = action_prob
        #self.action_prob = self.get_action_prob(idxs_remove_mask)

        #print('NUM POINTS REMOVED: {} of {}'.format(np.sum(idxs_remove_mask), idxs_in_box.shape))

        # Run pipeline
        detections = self.pipeline.run_detection([new_data_dict])
        tracks = self.pipeline.run_tracking(detections['results'], batch_mode=False)
        pred_token = self.target_instance + '_' + sample_token
        predictions = self.pipeline.run_prediction(tracks, [pred_token])

        # Log
        det_boxes = EvalBoxes.deserialize(detections['results'], DetectionBox)
        self.detections.append(det_boxes)
        self.tracks.append(tracks)
        prediction_list = [Prediction.deserialize(p) for p in predictions]
        self.predictions.append(prediction_list)
        self.log()
        self.step += 1

        return self.s0

    def reset(self, s0):
        # Reset pipeline
        self.step = s0
        self.action = None
        self.detections = []
        self.tracks = []
        self.predictions = []
        self.pipeline.reset()
        self.beta = BetaRadomization(self.fog_density, 0)
        # Simulate up to evaluation
        #print('--RESET--')

        ## TODO FIX RESET
        # for sample in self.base_detections['results'].keys():
        #     sample_det = {sample: self.base_detections['results'][sample]}
        #     self.pipeline.run_tracking(sample_det, batch_mode=False)

        return s0

    def is_goal(self):
        if len(self.predictions[-1]) == 0 and self.no_pred_failure:
            return True
        
        # If there is a prediction, check if avg error greater than threshold
        goal = False
        for pred in self.predictions[-1]:
            gt_pred = self.sample_to_gt_pred[pred.sample]
            sample_metrics = compute_prediction_metrics(pred, gt_pred, self.predict_eval_config)

            eval_idx = self.eval_k // 5
            eval_value = sample_metrics[self.eval_metric][self.target_instance][0][eval_idx]
            minadek = sample_metrics['MinADEK'][self.target_instance][0][-1]
            minade5 = sample_metrics['MinADEK'][self.target_instance][0][-2]
            minfde = sample_metrics['MinFDEK'][self.target_instance][0][-1]
            missrate = sample_metrics['MissRateTopK_2'][self.target_instance]

            print('{} value: {}'.format(self.eval_metric, eval_value))
            if eval_value > self.eval_metric_th:
                goal = True
        
        return goal

    def is_terminal(self):
        return self.step >= self.horizon

    def log(self):
        pass

    def observation(self):
        return self.s0

    def get_action_prob(self, ptrue, action):
        #n_true = np.sum(mask)
        #n_false = np.shape(mask)[0] - n_true
        return np.prod([p**action[i] * (1-p)**action[i] for i,p in enumerate(ptrue)])
        #return self.drop_likelihood**n_true * (1-self.drop_likelihood)**n_false
        #return np.prod()

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