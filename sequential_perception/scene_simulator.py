import time
from typing import Any, Dict, List
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

from sequential_perception.classical_pipeline import  PerceptionPipeline
from sequential_perception.evaluation import compute_prediction_metrics, load_sample_gt
from sequential_perception.input_representation_tracks import StaticLayerFromTrackingBW, AgentBoxesFromTrackingPost
from sequential_perception.pcdet_utils import  update_data_dict
from sequential_perception.nuscenes_utils import get_ordered_samples, render_box_with_pc, vis_sample_pointcloud
from sequential_perception.disturbances import BetaRadomization, haze_point_cloud
from sequential_perception.lisa import LISA
from sequential_perception.predict_helper_tracks import TrackingResultsPredictHelper


class ScenePerceptionSimulator:
    def __init__(self,
                 nuscenes: NuScenes,
                 pipeline: PerceptionPipeline,
                 scene_infos_list,
                 pipeline_config: Dict = {},
                 scene_token: str = 'fcbccedd61424f1b85dcbf8f897f9754',
                 warmup_steps=4,
                 weather_mode=1,
                 density = 5.0,
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
        self.density_probs = [0.6, 0.3, 0.1]
        self.scatter_fraction = 0.05
        self.density = density
        self.lisa = LISA()

        self.s0 = 0
        self.eval_idx = self.s0
        self.step = 0
        self.action = None
        self.action_mag = 0
        self.info = []
        self.best_reward = -np.inf

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
        assert len(ordered_samples) > 30

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

                            if ann_future.shape[0] >= 12:
                                pred_token = inst_record['token'] + '_' + ann_record['sample_token']

                                pred_candidate_info[pred_token] = ann_future

                        except:
                            pass
        
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

            if eval_value < self.eval_metric_th:
                passing_pred_tokens.append(pred_token)

        self.passing_pred_tokens = passing_pred_tokens

        # get unique samples where we want to run evaluation (find latest sample)
        token_sample_idxs = [ordered_samples.index(t.split('_')[1]) for t in self.passing_pred_tokens]
        self.last_sample_idx = np.max(token_sample_idxs)
        self.first_sample_idx = np.min(token_sample_idxs) - 3
        self.first_sample_idx = np.max([self.first_sample_idx, 1])

        self.horizon = (self.last_sample_idx - self.first_sample_idx) + 1

        self.set_density = False
        self.density = density
        if type(density) is list:
            self.set_density = True
            self.horizon += 1
            self.sim_density = density[0]
            assert len(density) == 3
        else:
            self.sim_density = density
        
        self.beta = BetaRadomization(self.sim_density, 0)
        self.ordered_sample_tokens = ordered_samples

        # Map sample tokens to lists of pred tokens
        sample2predtokens = {}
        for sample_token in self.ordered_sample_tokens:
            pred_tokens = [t for t in self.passing_pred_tokens if sample_token in t]
            sample2predtokens[sample_token] = pred_tokens

        self.sample2predtokens = sample2predtokens
        self.ordered_sample_tokens = self.ordered_sample_tokens[self.first_sample_idx-1:self.last_sample_idx]

        self.reset(self.s0)


    def simulate(self, actions: List[int], s0: int, render=False):
        path_length = 0
        self.reset(s0)
        self.info = []
        simulation_horizon = np.minimum(self.horizon, len(actions))
        while path_length < simulation_horizon:
            
            self.action = actions[path_length]
            self.step_simulation(self.action)
            if self.is_goal():
                return path_length, np.array(self.info)

            path_length += 1

        self.terminal = True
        return -1, np.array(self.info)

    def set_density(self, action: int):
        # DO NOT advance sim time
        rng = np.random.default_rng(action)
        density_idx = rng.choice(3, p=self.density_probs)
        self.sim_density = self.density[density_idx]
        self.beta = BetaRadomization(self.sim_density, 0)
        self.action_prob = self.density_probs[density_idx]
        self.log_action_prob = np.log(self.density_probs[density_idx])
        return

    def step_simulation(self, action: int):
        idx = self.step
        self.action = action
        sample_token = self.ordered_sample_tokens[idx] #  sample_token = self.eval_samples[idx]

        if self.sim_density == None:
            self.set_density(self.action)
            self.log()
            return self.s0

        data_dict = self.pcdet_infos[sample_token]
        points = data_dict['points']
        if self.weather_mode == 1:
            new_points, log_action_prob = self.lisa.haze_point_cloud(points, self.density, action)
        elif self.weather_mode == 2:
            new_points, log_action_prob = self.lisa.lisa_mc(points, self.density, action)
        elif self.weather_mode ==3:
            new_points, log_action_prob = self.lisa.lisa_avg(points, self.density, action)

        self.action_mag = points.shape[0] - new_points.shape[0] + np.sum(new_points[:, -1] > 0)

        self.new_points = new_points
        new_data_dict = update_data_dict(self.pipeline.detector.pcdet_dataset, data_dict, new_points[:, :5])

        self.log_action_prob = log_action_prob
        
        # Run pipeline
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
        if self.set_density:
            self.sim_density = None
        self.cnt = 0
        self.action_prob = 1.0
        self.log_action_prob = 0.0
        self.action_mag = 0

        self.sim_log = {'actions': [],
                'action_magnitudes': [],
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
                self.failure_eval_metrics.append(eval_value)
                self.failure_instances.append(pred_token.split('_')[0])
                self.failure_sample = pred_token.split('_')[1]
        
        return goal

    def is_terminal(self):
        return self.step >= self.horizon

    def log(self):
        self.sim_log['actions'].append(self.action)
        self.sim_log['action_magnitudes'].append(self.action_mag)
        self.sim_log['reward'] += self.log_action_prob
        if self.is_goal():
            self.sim_log['failure_sample'] = self.failure_sample
            self.sim_log['failure_agents'] = self.failure_instances
            self.sim_log['failure_metrics'] = self.failure_eval_metrics
            
            if self.sim_log['reward'] > self.best_reward:
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
        points = self.new_points
        ax = vis_sample_pointcloud(self.nuscenes,
                                   sample_token,
                                   gt_boxes=gt_boxes,
                                   pred_boxes=det_boxes,
                                   # pc=self.pcdet_infos[sample]['points'][self.sample_to_point_idxs[sample], :].T,
                                   pc=points.T,
                                   savepath=None)

        plt.sca(ax)
        scattered_points = points[points[:, -1] == 2, :]
        ax.scatter(scattered_points[:, 0],
                   scattered_points[:, 1],
                   c='r',
                   s=0.2)

        range_points = points[points[:, -1] == 1, :]
        ax.scatter(range_points[:, 0],
                   range_points[:, 1],
                   c='b',
                   s=0.2)

        savepath = render_path + str(self.step) + '_'+ sample_token
        plt.savefig(savepath, bbox_inches='tight')
        return ax


    def _render_prediction(self, render_path):
        import matplotlib.cm
        from nuscenes.eval.prediction.metrics import stack_ground_truth, mean_distances

        if len(self.predictions[-1]) == 0:
            return
        res = 0.1/4
        track_helper = TrackingResultsPredictHelper(self.nuscenes, self.tracks[-1]['results'])
        agent_rasterizer = AgentBoxesFromTrackingPost(track_helper, resolution=res, seconds_of_history=self.pipeline.predictor.seconds_of_history)
        map_rasterizer = StaticLayerFromTrackingBW(track_helper, resolution=res)
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
            cmap = matplotlib.cm.get_cmap('Blues', 7)
            msize=7
            lwidth = 2.0
            fig, ax = plt.subplots(1, 1, figsize=(9, 9))
            plt.imshow(input_image)

            if 'ADE' in self.eval_metric:
                eval_arr = mean_dists
            else:
                eval_arr = final_dists

            for i in range(5):
                traj = output_trajs[i]
                label = 'FDE={:4.2f}m'.format(eval_arr[i])
                plt.plot((1/res)*traj[:, 0] + 25/res, -((1/res)*traj[:, 1]) + 40/res, color=cmap(6-i), marker='o', markersize=msize, label=label, linewidth=lwidth, zorder=10-i)

            plt.plot((1/res)*gt_future_local[:, 0] + 25/res, -((1/res)*gt_future_local[:, 1]) + 40/res, color='black', marker='o', markersize=msize, linewidth=lwidth, label='Ground Truth')
            leg = plt.legend(frameon=True, facecolor='white', framealpha=1, prop={'size': 26}, loc='center left', bbox_to_anchor=(1, 0.5))
            plt.xlim(0, 50/res)
            plt.ylim(50/res, 0)
            # Hide grid lines
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
                    
            plt.savefig(savepath, bbox_inches='tight', dpi=300)
            plt.savefig(savepath + '.pdf', bbox_inches='tight', dpi=300)
        plt.close('all')