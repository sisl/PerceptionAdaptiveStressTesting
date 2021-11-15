from random import sample
import time
from typing import Any, Dict, List
from unicodedata import category
import numpy as np
from numpy.core.fromnumeric import cumsum
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

from sequential_perception.classical_pipeline import  PerceptionPipeline
from sequential_perception.evaluation import compute_prediction_metrics, load_sample_gt
from sequential_perception.input_representation_tracks import AgentBoxesFromTracking, StaticLayerFromTracking
from sequential_perception.pcdet_utils import get_boxes_for_pcdet_data, update_data_dict
from sequential_perception.nuscenes_utils import get_ordered_samples, render_box_with_pc, vis_sample_pointcloud
from sequential_perception.disturbances import BetaRadomization, haze_point_cloud
from sequential_perception.predict_helper_tracks import TrackingResultsPredictHelper

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

        # Load all PCDet data for scene
        pcdet_dataset = pipeline.detector.pcdet_dataset
        self.pcdet_infos = {}
        # for data_dict in pcdet_dataset:
        #     if data_dict['metadata']['token'] in ordered_samples:
        #         self.pcdet_infos[data_dict['metadata']['token']] = data_dict
        print(len(pcdet_dataset))
        for i in range(len(pcdet_dataset)):
            data_dict = pcdet_dataset[i]
            if data_dict['metadata']['token'] in ordered_samples:
                self.pcdet_infos[data_dict['metadata']['token']] = data_dict

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
        

        #scene_data_dicts = [self.pcdet_dataset[i] for i in range(len(self.pcdet_dataset))]
        scene_data_dicts = list(self.pcdet_infos.values())
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
        sample_token = self.ordered_sample_tokens[idx] #  sample_token = self.eval_samples[idx]
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
        new_data_dict = update_data_dict(self.pipeline.detector.pcdet_dataset, data_dict, new_points[:, :5])
        self.action_prob = action_prob
        #self.action_prob = self.get_action_prob(idxs_remove_mask)

        #print('NUM POINTS REMOVED: {} of {}'.format(np.sum(idxs_remove_mask), idxs_in_box.shape))

        # Run pipeline
        detections = self.pipeline.run_detection([new_data_dict])
        tracks = self.pipeline.run_tracking(detections['results'], batch_mode=False)
        
        
        #pred_token = self.target_instance + '_' + sample_token
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
        # prediction_list = [Prediction.deserialize(p) for p in predictions]
        # self.predictions.append(prediction_list)
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
        self.cnt = 0
        self.action_prob = 1
        # Simulate up to evaluation
        #print('--RESET--')

        ## TODO FIX RESET
        #Simulate up to first sample index
        # for i in range(self.first_sample_idx):
        #     sample_token = self.ordered_sample_tokens[i]
        # for sample in self.base_detections['results'].keys():
        #     sample_det = {sample: self.base_detections['results'][sample]}
        #     self.pipeline.run_tracking(sample_det, batch_mode=False)

        return s0

    def is_goal(self):
        # if len(self.predictions[-1]) == 0 and self.no_pred_failure:
        #     return True
        if self.step == 0:
            return False 

        sample_token = self.ordered_sample_tokens[self.step-1]
        num_preds = len([p for p in self.predictions[-1] if p])
        if num_preds != len(self.sample2predtokens[sample_token]) and self.no_pred_failure:
            return True
        
        # If there is a prediction, check if avg error greater than threshold
        goal = False
        for pred in self.predictions[-1]:
            #self.cnt += 1
            #gt_pred = self.sample_to_gt_pred[pred.sample]
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


    def _render_detection(self, render_path):
        # from nuscenes.eval.detection.render import visualize_sample
        # from nuscenes.eval.common.loaders import load_gt

        # If no detections, do nothing
        if len(self.detections) == 0:
            return

        # render_path = '/scratch/hdelecki/ford/output/ast/pointdrop/plots/'
        
        #render_path = '/mnt/hdd/ford_ws/output/ast/plots/test/'
        # render_path = plot_path

        det_boxes = self.detections[-1]
        sample_token = det_boxes.sample_tokens[0]
        sample_rec = self.nuscenes.get('sample', sample_token)
        sd_record = self.nuscenes.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        cs_record = self.nuscenes.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        pose_record = self.nuscenes.get('ego_pose', sd_record['ego_pose_token'])
        #det_boxes = EvalBoxes.deserialize(detections['results'], DetectionBox)
        # gt_boxes = load_gt(self.nuscenes, 'mini_val', DetectionBox)
        gt_boxes = load_sample_gt(self.nuscenes, sample_token, DetectionBox)
        #gt_boxes_list = boxes_to_sensor(gt_boxes[sample_token], pose_record, cs_record)

        # for box in gt_boxes_list:
        #     if box.token == self.target_instance:
        #         gt_box = box
        #         break

        # savepath = render_path + str(self.step) + '_' + sample_token
        # points = self.pcdet_infos[sample_token]['points'],

        # idxs_in_box = self.sample_to_point_idxs[sample_token]
        # idxs_remove_mask = bernoulli.rvs(self.drop_likelihood,
        #                                     size=idxs_in_box.shape[0],
        #                                     random_state=self.action)
        # idxs_to_remove = idxs_in_box[idxs_remove_mask>0]
        # new_points = np.delete(points, idxs_to_remove, axis=0)
        #points = self.get_pointcloud(sample_token, self.action)
        #points = self.pcdet_infos[sample_token]['points']
        #print(points.shape)
        points = self.new_points

        ax = vis_sample_pointcloud(self.nuscenes,
                                   sample_token,
                                   gt_boxes=gt_boxes,
                                   pred_boxes=det_boxes,
                                   # pc=self.pcdet_infos[sample]['points'][self.sample_to_point_idxs[sample], :].T,
                                   pc=points.T,
                                   savepath=None)
        # ax = render_box_with_pc(self.nuscenes,
        #                         self.eval_samples[self.step-1],
        #                         self.target_instance,
        #                         points[:, :4].T,
        #                         pred_boxes=self.detections[self.step-1][sample_token],
        #                         margin=6)
        
        # highlight removed points
        plt.sca(ax)
        # idxs_in_box = self.sample_to_point_idxs[sample_token]
        # idxs_remove_mask = bernoulli.rvs(self.drop_likelihood,
        #                                     size=idxs_in_box.shape[0],
        #                                     random_state=self.action)
        # idxs_to_remove = idxs_in_box[idxs_remove_mask>0]
        scattered_points = points[points[:, -1] == 2, :]
        ax.scatter(scattered_points[:, 0],
                   scattered_points[:, 1],
                   c='r',
                   s=0.2)


        #render_path = '/scratch/hdelecki/ford/output/ast/pointdrop/plots/'
        savepath = render_path + str(self.step) + '_'+ sample_token
        #plt.set_cmap('viridis')
        plt.savefig(savepath, bbox_inches='tight')

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

            #input_image = input_representation.make_input_representation(inst_token, sample_token)

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


            #input_image = self.pipeline.predictor.input_image
            #output_trajs = self.pipeline.predictor.output_trajs
            #output_probs = self.pipeline.predictor.output_probs
            # print(output_probs)

            #gt_future = self.sample_to_gt_pred[sample_token]
            pred_token = prediction.instance + '_' + prediction.sample
            gt_future = self.pred_candidate_info[pred_token]
            
            
            # predicted_track = self.pipeline.predictor.predicted_track
            predicted_track = nearest_track
            gt_future_local = convert_global_coords_to_local(gt_future, predicted_track['translation'], predicted_track['rotation'])

            stacked_ground_truth = stack_ground_truth(gt_future_local, prediction.number_of_modes)
            mean_dists = mean_distances(np.array(output_trajs), stacked_ground_truth).flatten()
            final_dists = final_distances(np.array(output_trajs), stacked_ground_truth).flatten()
            dist_sorted_idxs = np.argsort(mean_dists)

            # render_path = '/mnt/hdd/ford_ws/output/ast/plots/test/'
            savepath = render_path + str(self.step) + '_' + inst_token + '_' + sample_token

            #colors = ['tab1', 'tab2', 'tab3', 'tab4', 'tab5', 'tab6', 'tab7', 'tab8', 'tab9', 'tab10']
            cmap = matplotlib.cm.get_cmap('tab10')
            msize=7
            lwidth = 1.4
            fig, ax = plt.subplots(1, 1, figsize=(9, 9))
            plt.imshow(input_image)


            if 'ADE' in self.eval_metric:
                eval_arr = mean_dists
            else:
                eval_arr = final_dists
            #for traj in output_trajs:
            for i in range(5):
                traj = output_trajs[i]
                #plt.scatter(10.*traj[:, 0] + 250, -(10.*traj[:, 1]) + 400, color='green', s=10, alpha=output_probs[i])
                #plt.scatter(10.*traj[:, 0] + 250, -(10.*traj[:, 1]) + 400, color=colors[i], s=10)
                # label = 'p={:4.3f}'.format(output_probs[i])
                label = 'p={:4.3f}, {}{}={:4.2f}'.format(output_probs[i], self.eval_metric[3:-1], self.eval_k, eval_arr[i])
                plt.plot(10.*traj[:, 0] + 250, -(10.*traj[:, 1]) + 400, color=cmap(i), marker='o', markersize=msize, label=label, zorder=10-i)

            # Plot top 5 closest predicted
            # for i in range(5):
            #     idx = dist_sorted_idxs[i]
            #     traj = output_trajs[idx]
            #     #plt.scatter(10.*traj[:, 0] + 250, -(10.*traj[:, 1]) + 400, color='green', s=10, alpha=output_probs[i])
            #     #plt.scatter(10.*traj[:, 0] + 250, -(10.*traj[:, 1]) + 400, color=colors[i], s=10)
            #     label = 'P={:4.3f}, ADE={:4.2f}'.format(output_probs[idx], mean_dists[idx])
            #     plt.plot(10.*traj[:, 0] + 250, -(10.*traj[:, 1]) + 400, color=cmap(i), marker='o', markersize=msize, linewidth=lwidth, label=label, zorder=10-i)

            plt.plot(10.*gt_future_local[:, 0] + 250, -(10.*gt_future_local[:, 1]) + 400, color='black', marker='o', markersize=msize, linewidth=lwidth, label='GT')
            #leg = plt.legend(facecolor='white', framealpha=1, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            leg = plt.legend(frameon=True, facecolor='white', framealpha=1, prop={'size': 14})
            plt.xlim(0, 500)
            plt.ylim(500, 0)
            # Hide grid lines
            ax.grid(False)

            # Hide axes ticks
            ax.set_xticks([])
            ax.set_yticks([])

            # plt.sca(ax)
            # plt.show()
                    
            plt.savefig(savepath, dpi=600)
            #plt.savefig(savepath, bbox_inches='tight')
        plt.close('all')
        #return ax