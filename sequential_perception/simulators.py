
from random import sample
import time
from typing import Any, Dict, List
import numpy as np
from scipy.stats import bernoulli
import matplotlib.pyplot as plt

from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.prediction.config import load_prediction_config
from nuscenes.eval.prediction.data_classes import Prediction
from nuscenes.prediction.helper import PredictHelper, convert_global_coords_to_local
from nuscenes.utils.geometry_utils import points_in_box, view_points
from sequential_perception import input_representation_tracks

from sequential_perception.classical_pipeline import ClassicalPerceptionPipeline, PerceptionPipeline
from sequential_perception.evaluation import PipelineEvaluation, compute_prediction_metrics, load_sample_gt
from sequential_perception.input_representation_tracks import AgentBoxesFromTrackingPost, StaticLayerFromTrackingBW
from sequential_perception.pcdet_utils import get_boxes_for_pcdet_data, update_data_dict
from sequential_perception.nuscenes_utils import get_ordered_samples, render_box_with_pc, vis_sample_pointcloud
from sequential_perception.disturbances import BetaRadomization, haze_point_cloud
from sequential_perception.predict_helper_tracks import TrackingResultsPredictHelper
from nuscenes.prediction.input_representation.combinators import Rasterizer
from nuscenes.prediction.input_representation.interface import InputRepresentation
from pyquaternion import Quaternion

class ClosedLoopPointDropPerceptionSimulator:
    def __init__(self,
                 nuscenes: NuScenes,
                 pipeline: ClassicalPerceptionPipeline,
                 pipeline_config: Dict = {},
                 scene_token: str = 'fcbccedd61424f1b85dcbf8f897f9754',
                 target_instance: str = '045cd82a77a1472499e8c15100cb5ff3',
                 eval_samples: List = [],
                 drop_likelihood = 0.1,
                 max_range = 25,
                 eval_metric = 'MinADEK',
                 eval_k = 10,
                 eval_metric_th=5.0,
                 no_pred_failure = False) -> None:

        assert eval_metric in ['MinADEK', 'MinFDEK']
        assert eval_k in [1, 5, 10]

        self.nuscenes = nuscenes
        self.pipeline = pipeline
        self.pipeline_config = pipeline_config
        self.drop_likelihood = drop_likelihood
        self.no_pred_failure = no_pred_failure # Fail if no prediction during an eval sample?
        self._action = None
        self.target_instance = target_instance
        self.predict_eval_config_name = 'predict_2020_icra.json'
        # Get all samples for scene
        ordered_samples = get_ordered_samples(nuscenes, scene_token)

        # find samples where instance is active
        target_ann_tokens = set(nuscenes.field2token('sample_annotation', 'instance_token', target_instance))
        target_annotations = [nuscenes.get('sample_annotation', t) for t in target_ann_tokens]
        target_annotations = sorted(target_annotations, key=lambda x: ordered_samples.index(x['sample_token']))
        
        last_target_sample_token = target_annotations[-1]['sample_token']
        ordered_samples = ordered_samples[:ordered_samples.index(last_target_sample_token)]
        self.ordered_samples = ordered_samples[:len(ordered_samples)-12]
        target_annotations = [ann for ann in target_annotations if ann['sample_token'] in self.ordered_samples]
    
        # filter samples for when target instance is within in ~20 m
        filtered_target_annotations = []
        for idx, sample_ann in enumerate(target_annotations):
            sample_token = sample_ann['sample_token']
            sample_rec = nuscenes.get('sample', sample_token)
            sd_record = nuscenes.get('sample_data', sample_rec['data']['LIDAR_TOP'])
            pose_record = nuscenes.get('ego_pose', sd_record['ego_pose_token'])

            # Both boxes and ego pose are given in global coord system, so distance can be calculated directly.
            box = nuscenes.get_box(sample_ann['token'])
            ego_translation = (box.center[0] - pose_record['translation'][0],
                               box.center[1] - pose_record['translation'][1],
                               box.center[2] - pose_record['translation'][2])
            if np.linalg.norm(ego_translation) <= max_range:
                filtered_target_annotations.append(sample_ann)

        self.target_annotations = filtered_target_annotations
        self.target_samples = [ann['sample_token'] for ann in self.target_annotations]

        assert len(self.target_samples) > 2, \
               'At least 2 valid samples required, found {}'.format(len(self.target_samples))

        if len(eval_samples) == 0:
            self.eval_samples = self.target_samples
            self.eval_annotations = self.target_annotations
        else:
            self.eval_samples = eval_samples
            self.eval_annotations = [ann for ann in self.target_annotations if ann['sample_token'] in self.eval_samples]

        # Load all PCDet data for scene
        pcdet_dataset = pipeline.pcdet_dataset
        self.pcdet_infos = {}
        for data_dict in pcdet_dataset:
            if data_dict['metadata']['token'] in self.ordered_samples:
                self.pcdet_infos[data_dict['metadata']['token']] = data_dict

        # Get boxes for target instance
        self.sample_to_point_idxs = {}
        self.sample_to_boxes = {}
        self.sample_to_gt_pred = {}
        nusc_helper = PredictHelper(self.nuscenes)
        self.predict_eval_config = load_prediction_config(nusc_helper, self.predict_eval_config_name)
        for ann in self.eval_annotations:
            data_dict = self.pcdet_infos[ann['sample_token']]
            boxes = get_boxes_for_pcdet_data(self.nuscenes, data_dict, ann)
            self.sample_to_boxes[ann['sample_token']] = boxes
        
            points = data_dict['points'] # [n x 5]
            masks = [points_in_box(b, points[:, :3].T, wlh_factor=1) for b in boxes]
            overall_mask = np.logical_or.reduce(np.array(masks))
            idxs_in_box = overall_mask.nonzero()[0]
            self.sample_to_point_idxs[ann['sample_token']] = idxs_in_box

            # Run detection for samples outside eval_samples 
            base_data_dicts = [d for key,d in self.pcdet_infos.items() if key not in self.eval_samples]
            self.base_detections = self.pipeline.run_detection(base_data_dicts)

            # Pre-calculate ground truth trajectory predictions
            sample_future = nusc_helper.get_future_for_agent(self.target_instance,
                                                             ann['sample_token'],
                                                             seconds=6.0,
                                                             in_agent_frame=False)
            self.sample_to_gt_pred[ann['sample_token']] = sample_future

        # Get predict tokens
        self.predict_tokens = self.get_predict_tokens()

        self.s0 = 0
        self.eval_idx = self.s0
        self.horizon = len(self.eval_samples)
        self.step = 0
        self.action = None
        self.info = []

        self.eval_metric = eval_metric
        self.eval_k = eval_k
        self.eval_metric_th = eval_metric_th
        
        self.detections = []
        self.tracks = []
        self.predictions = []

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

    def step_simulation(self, action: int):
        idx = self.step
        self.action = action
        sample_token = self.eval_samples[idx]
        print('STEP: {}'.format(self.step))

        data_dict = self.pcdet_infos[sample_token]
        points = data_dict['points']
        idxs_in_box = self.sample_to_point_idxs[sample_token]

        # Select point to remove
        idxs_remove_mask = bernoulli.rvs(self.drop_likelihood,
                                            size=idxs_in_box.shape[0],
                                            random_state=action)
        self.action_prob = self.get_action_prob(idxs_remove_mask)
        idxs_to_remove = idxs_in_box[idxs_remove_mask>0]

        # Remove points and update detection input
        new_points = np.delete(points, idxs_to_remove, axis=0)
        new_data_dict = update_data_dict(self.pipeline.pcdet_dataset, data_dict, new_points)

        print('NUM POINTS REMOVED: {} of {}'.format(np.sum(idxs_remove_mask), idxs_in_box.shape))

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

        # Simulate up to evaluation
        for sample in self.base_detections['results'].keys():
            sample_det = {sample: self.base_detections['results'][sample]}
            self.pipeline.run_tracking(sample_det, batch_mode=False)

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

    def get_action_prob(self, mask):
        n_true = np.sum(mask)
        n_false = np.shape(mask)[0] - n_true
        return np.prod([self.drop_likelihood**x * (1-self.drop_likelihood)**x for x in mask])

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

    def render(self, render_path=None):
        det_ax = self._render_detection(render_path=render_path)
        pred_ax = self._render_prediction(render_path=render_path)
        self._render_disturbance(render_path=render_path)
        return det_ax, pred_ax
        
    def _render_detection(self, render_path=None):
        # If no detections, do nothing
        if len(self.detections) == 0:
            return

        det_boxes = self.detections[-1]
        sample_token = det_boxes.sample_tokens[0]
        gt_boxes = load_sample_gt(self.nuscenes, sample_token, DetectionBox)

        savepath = render_path + str(self.step) + '_' + sample_token
        points = self.pcdet_infos[sample_token]['points']

        ax = vis_sample_pointcloud(self.nuscenes,
                                   sample_token,
                                   gt_boxes=gt_boxes,
                                   pred_boxes=det_boxes,
                                   # pc=self.pcdet_infos[sample]['points'][self.sample_to_point_idxs[sample], :].T,
                                   pc=points.T,
                                   savepath=None)
        nusc = self.nuscenes
        sample_record = nusc.get('sample', sample_token)
        for anntoken in sample_record['anns']:
            ann_record = nusc.get('sample_annotation', anntoken)
            if ann_record['instance_token'] == self.target_instance:
                break

        sample_rec = nusc.get('sample', sample_token)
        sd_record = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
        gt_box = nusc.get_box(ann_record['token'])
        gt_box.translate(-np.array(pose_record['translation']))
        gt_box.rotate(Quaternion(pose_record['rotation']).inverse)

        #  Move box to sensor coord system.
        margin = 12
        gt_box.translate(-np.array(cs_record['translation']))
        gt_box.rotate(Quaternion(cs_record['rotation']).inverse)
        corners = view_points(gt_box.corners(), np.eye(3), False)[:2, :]
        ax.set_xlim([np.min(corners[0, :]) - margin, np.max(corners[0, :]) + margin])
        ax.set_ylim([np.min(corners[1, :]) - margin, np.max(corners[1, :]) + margin])
        
        # highlight removed points
        plt.sca(ax)
        idxs_in_box = self.sample_to_point_idxs[sample_token]
        idxs_remove_mask = bernoulli.rvs(self.drop_likelihood,
                                            size=idxs_in_box.shape[0],
                                            random_state=self.action)
        idxs_to_remove = idxs_in_box[idxs_remove_mask>0]
        ax.scatter(points[idxs_to_remove, 0],
                   points[idxs_to_remove, 1],
                   c='r',
                   s=3)
        ax.set_title('')
        savepath = render_path + str(self.step) + '_'+ sample_token
        plt.savefig(savepath + '.pdf', bbox_inches='tight', dpi=300)
        plt.savefig(savepath + '.png', bbox_inches='tight', dpi=300)

        return ax

    def _render_prediction(self, render_path=None):
        import matplotlib.cm
        from nuscenes.eval.prediction.metrics import stack_ground_truth, mean_distances, final_distances

        if len(self.predictions[-1]) == 0:
            return

        prediction = self.predictions[-1][0]
        sample_token = prediction.sample
        inst_token = prediction.instance
        output_trajs = self.pipeline.predictor.output_trajs
        output_probs = self.pipeline.predictor.output_probs

        res = 0.1/4
        track_helper = TrackingResultsPredictHelper(self.nuscenes, self.tracks[-1]['results'])
        agent_rasterizer = AgentBoxesFromTrackingPost(track_helper, resolution=res, seconds_of_history=self.pipeline.predictor.seconds_of_history)
        map_rasterizer = StaticLayerFromTrackingBW(track_helper, resolution=res)
        input_representation = InputRepresentation(map_rasterizer, agent_rasterizer, Rasterizer())
        nusc_helper = PredictHelper(self.nuscenes)

        gt_future = self.sample_to_gt_pred[sample_token]
        predicted_track = self.pipeline.predictor.predicted_track
        gt_future_local = convert_global_coords_to_local(gt_future, predicted_track['translation'], predicted_track['rotation'])

        stacked_ground_truth = stack_ground_truth(gt_future_local, prediction.number_of_modes)
        mean_dists = mean_distances(np.array(output_trajs), stacked_ground_truth).flatten()
        final_dists = final_distances(np.array(output_trajs), stacked_ground_truth).flatten()
        dist_sorted_idxs = np.argsort(mean_dists)

        ## make input image
        nearest_track = self.pipeline.predictor._find_nearest_track(track_helper, inst_token, sample_token)
        if nearest_track == None:
            return
        track_instance = nearest_track['tracking_id']

        input_image = input_representation.make_input_representation(track_instance, sample_token)
        savepath = render_path + str(self.step) + '_' + inst_token + '_' + sample_token
        cmap = matplotlib.cm.get_cmap('Blues', 7)
        msize=7
        lwidth = 2
        fig, ax = plt.subplots(1, 1, figsize=(9, 9))
        plt.imshow(input_image)
        for i in range(1):
            traj = output_trajs[i]
            label = 'FDE={:4.2f}m'.format(final_dists[i])
            plt.plot((1/res)*traj[:, 0] + 25/res, -((1/res)*traj[:, 1]) + 40/res, color=cmap(5), marker='o', markersize=msize, label=label, linewidth=lwidth, zorder=10-i)
        plt.plot((1/res)*gt_future_local[:, 0] + 25/res, -((1/res)*gt_future_local[:, 1]) + 40/res, color='black', marker='o', markersize=msize, linewidth=lwidth, label='Ground Truth')
        leg = plt.legend(frameon=True, facecolor='white', framealpha=1, prop={'size': 26})
        plt.xlim(0, 50/res)
        plt.ylim(50/res, 0)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.sca(ax)
        plt.savefig(savepath + '.pdf', bbox_inches='tight', dpi=400)
        plt.savefig(savepath + '.png', bbox_inches='tight', dpi=400)
        plt.close('all')
        return ax

    def _render_disturbance(self, render_path=None):

        sample_token = self.eval_samples[self.step-1]
        target_box = self.sample_to_boxes[sample_token][0]
        idxs_in_box = self.sample_to_point_idxs[sample_token]
        points = self.pcdet_infos[sample_token]['points']
        points_in_box = points[idxs_in_box]
        idxs_remove_mask = bernoulli.rvs(self.drop_likelihood,
                                        size=idxs_in_box.shape[0],
                                        random_state=self.action)
        idxs_to_remove = idxs_in_box[idxs_remove_mask>0]

        dummy = [0.0, 0.0, -100., 1., 1.]
        points_in_box = np.vstack([points_in_box, dummy])

        quat = target_box.orientation
        inv_rot = quat.inverse.rotation_matrix

        side_rot = np.array([[1, 0, 0],
                                [0, 0, 1],
                                [0, -1, 0]])

        savepath = render_path + str(self.step) + '_action_side_' + sample_token

        ax = render_box_with_pc(self.nuscenes,
                                sample_token,
                                self.target_instance,
                                points_in_box[:, :4].T,
                                pred_boxes=None,
                                margin=1,
                                view=side_rot@inv_rot)

        removed_points = points[idxs_to_remove, :3]
        removed_points_view = view_points(removed_points.T, side_rot@inv_rot, False)             
        ax.scatter(removed_points_view[0, :],
                   removed_points_view[1, :],
                   c='r',
                   s=3)

        plt.sca(ax)
        plt.savefig(savepath, bbox_inches='tight')


        front_rot = np.array([[0, 0, -1],
                                [0, 1, 0],
                                [1, 0, 0]])
        
        savepath = render_path + str(self.step) + '_action_front_' + sample_token
        ax = render_box_with_pc(self.nuscenes,
                                sample_token,
                                self.target_instance,
                                points_in_box[:, :4].T,
                                pred_boxes=None,
                                margin=1,
                                view=front_rot@side_rot@inv_rot)
        removed_points = points[idxs_to_remove, :3]
        removed_points_view = view_points(removed_points.T, front_rot@side_rot@inv_rot, False)             
        ax.scatter(removed_points_view[0, :],
                   removed_points_view[1, :],
                   c='r',
                   s=3)

        plt.sca(ax)
        plt.savefig(savepath, bbox_inches='tight')




def render_detections(self, detections):
    from nuscenes.eval.detection.render import visualize_sample
    from nuscenes.eval.common.loaders import load_gt
    render_path = '/scratch/hdelecki/ford/output/sim_testing/plots/'
    det_boxes = EvalBoxes.deserialize(detections['results'], DetectionBox)
    gt_boxes = load_gt(self.nuscenes, 'mini_val', DetectionBox)
    for sample in det_boxes.sample_tokens:
        if sample in self.eval_samples:
            visualize_sample(self.nuscenes,
                            sample,
                            gt_boxes=gt_boxes,
                            pred_boxes=det_boxes,
                            # pc=self.pcdet_infos[sample]['points'][self.sample_to_point_idxs[sample], :].T,
                            pc=self.pcdet_infos[sample]['points'].T,
                            savepath=render_path+sample)



def _find_nearest_track(nusc_helper, track_helper, gt_instance, sample):
    gt_annotation = nusc_helper.get_sample_annotation(gt_instance, sample)
    gt_pos = np.array(gt_annotation['translation'][:2])
    tracks = track_helper.get_annotations_for_sample(sample)
    min_dist = np.inf
    best_track = None
    for track in tracks:
        if track['tracking_name'] in ['car', 'bus', 'truck']:
            track_pos = np.array(track['translation'][:2])
            dist = np.linalg.norm(track_pos - gt_pos)
            if dist < min_dist:
                min_dist = dist
                best_track = track

    if min_dist > 2.0:
        print('Track could not be matched for instance={} and sample={}'.format(gt_instance, sample))
        best_track = None

    return best_track

class ClosedLoopPerceptionSimulator:
    def __init__(self,
                 nuscenes: NuScenes,
                 pipeline: ClassicalPerceptionPipeline,
                 pipeline_config: Dict = {},
                 scene_token: str = 'fcbccedd61424f1b85dcbf8f897f9754',
                 target_instance: str = '045cd82a77a1472499e8c15100cb5ff3',
                 eval_samples: List = [],
                 scatter_fraction = 0.05, # [-]
                 fog_density = 0.005, # 0.0 - 0.08 [1/m]
                 max_range = 25,
                 eval_metric = 'MinADEK',
                 eval_k = 10,
                 eval_metric_th=5.0,
                 no_pred_failure = False) -> None:

        assert eval_metric in ['MinADEK', 'MinFDEK']
        assert eval_k in [1, 5, 10]

        self.nuscenes = nuscenes
        self.pipeline = pipeline
        self.pipeline_config = pipeline_config
        self.no_pred_failure = no_pred_failure # Fail if no prediction during an eval sample?
        self._action = None
        self.target_instance = target_instance
        self.predict_eval_config_name = 'predict_2020_icra.json'

        self.beta = BetaRadomization(fog_density, 0)
        self.scatter_fraction = scatter_fraction
        self.fog_density = fog_density
        # Get all samples for scene
        ordered_samples = get_ordered_samples(nuscenes, scene_token)

        # find samples where instance is active
        target_ann_tokens = set(nuscenes.field2token('sample_annotation', 'instance_token', target_instance))
        target_annotations = [nuscenes.get('sample_annotation', t) for t in target_ann_tokens]
        target_annotations = sorted(target_annotations, key=lambda x: ordered_samples.index(x['sample_token']))
        
        last_target_sample_token = target_annotations[-1]['sample_token']
        ordered_samples = ordered_samples[:ordered_samples.index(last_target_sample_token)]
        self.ordered_samples = ordered_samples[:len(ordered_samples)-12]
        
        target_annotations = [ann for ann in target_annotations if ann['sample_token'] in self.ordered_samples]
    
        # filter samples for when target instance is within in ~20 m
        filtered_target_annotations = []
        for idx, sample_ann in enumerate(target_annotations):
            sample_token = sample_ann['sample_token']
            sample_rec = nuscenes.get('sample', sample_token)
            sd_record = nuscenes.get('sample_data', sample_rec['data']['LIDAR_TOP'])
            pose_record = nuscenes.get('ego_pose', sd_record['ego_pose_token'])


            # Both boxes and ego pose are given in global coord system, so distance can be calculated directly.
            box = nuscenes.get_box(sample_ann['token'])
            ego_translation = (box.center[0] - pose_record['translation'][0],
                               box.center[1] - pose_record['translation'][1],
                               box.center[2] - pose_record['translation'][2])
            if np.linalg.norm(ego_translation) <= max_range:
                filtered_target_annotations.append(sample_ann)

        self.target_annotations = filtered_target_annotations
        self.target_samples = [ann['sample_token'] for ann in self.target_annotations]

        assert len(self.target_samples) > 2, \
               'At least 2 valid samples required, found {}'.format(len(self.target_samples))

        if len(eval_samples) == 0:
            self.eval_samples = self.target_samples
            self.eval_annotations = self.target_annotations
        else:
            self.eval_samples = eval_samples
            self.eval_annotations = [ann for ann in self.target_annotations if ann['sample_token'] in self.eval_samples]


        [print(self.ordered_samples.index(t)) for t in self.eval_samples]

        # Load all PCDet data for scene
        pcdet_dataset = pipeline.pcdet_dataset
        self.pcdet_infos = {}
        for data_dict in pcdet_dataset:
            if data_dict['metadata']['token'] in self.ordered_samples:
                self.pcdet_infos[data_dict['metadata']['token']] = data_dict

        # Get boxes for target instance
        #sample_to_target_boxes = {}
        self.sample_to_point_idxs = {}
        self.sample_to_boxes = {}
        self.sample_to_gt_pred = {}
        nusc_helper = PredictHelper(self.nuscenes)
        self.predict_eval_config = load_prediction_config(nusc_helper, self.predict_eval_config_name)
        for ann in self.eval_annotations:
            data_dict = self.pcdet_infos[ann['sample_token']]
            boxes = get_boxes_for_pcdet_data(self.nuscenes, data_dict, ann)
            self.sample_to_boxes[ann['sample_token']] = boxes
        
            points = data_dict['points'] # [n x 5]

            # masks = [points_in_box(b, points[:, :3].T, wlh_factor=np.array([1.2, 1.2, 10])) for b in boxes]
            masks = [points_in_box(b, points[:, :3].T, wlh_factor=1) for b in boxes]
            overall_mask = np.logical_or.reduce(np.array(masks))
            idxs_in_box = overall_mask.nonzero()[0]

            self.sample_to_point_idxs[ann['sample_token']] = idxs_in_box


            # Run detection for samples outside eval_samples 
            base_data_dicts = [d for key,d in self.pcdet_infos.items() if key not in self.eval_samples]
            self.base_detections = self.pipeline.run_detection(base_data_dicts)

            # Pre-calculate ground truth trajectory predictions
            sample_future = nusc_helper.get_future_for_agent(self.target_instance,
                                                             ann['sample_token'],
                                                             seconds=6.0,
                                                             in_agent_frame=False)
            self.sample_to_gt_pred[ann['sample_token']] = sample_future

        # Get predict tokens
        self.predict_tokens = self.get_predict_tokens()

        self.s0 = 0
        self.eval_idx = self.s0
        self.horizon = len(self.eval_samples)
        self.step = 0
        self.action = None
        self.info = []

        self.eval_metric = eval_metric
        self.eval_k = eval_k
        self.eval_metric_th = eval_metric_th
        
        self.detections = []
        self.tracks = []
        self.predictions = []

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

    def step_simulation(self, action: int):
        idx = self.step
        self.action = action
        sample_token = self.eval_samples[idx]
        print('STEP: {}'.format(self.step))

        data_dict = self.pcdet_infos[sample_token]
        points = data_dict['points']
        new_points, action_prob = haze_point_cloud(points, self.beta, self.scatter_fraction, action)
        self.new_points = new_points
        new_data_dict = update_data_dict(self.pipeline.pcdet_dataset, data_dict, new_points[:, :5])
        self.action_prob = action_prob

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
        for sample in self.base_detections['results'].keys():
            sample_det = {sample: self.base_detections['results'][sample]}
            self.pipeline.run_tracking(sample_det, batch_mode=False)

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

    def render(self):
        det_ax = self._render_detection()
        pred_ax = self._render_prediction()
        return det_ax, pred_ax
        
    def _render_detection(self):
        if len(self.detections) == 0:
            return

        render_path = '/mnt/hdd/output/ford/ast/fog/plots/'
        det_boxes = self.detections[-1]
        sample_token = det_boxes.sample_tokens[0]
        gt_boxes = load_sample_gt(self.nuscenes, sample_token, DetectionBox)
        savepath = render_path + str(self.step) + '_' + sample_token
        points = self.new_points[:, :5]

        ax = vis_sample_pointcloud(self.nuscenes,
                                   sample_token,
                                   gt_boxes=gt_boxes,
                                   pred_boxes=det_boxes,
                                   # pc=self.pcdet_infos[sample]['points'][self.sample_to_point_idxs[sample], :].T,
                                   pc=points.T,
                                   savepath=savepath)
        plt.sca(ax)


        render_path = render_path = '/mnt/hdd/output/ford/ast/fog/plots/'
        savepath = render_path + str(self.step) + '_'+ sample_token
        plt.savefig(savepath, bbox_inches='tight')

        return ax

    def _render_prediction(self):
        import matplotlib.cm
        from nuscenes.eval.prediction.metrics import stack_ground_truth, mean_distances

        if len(self.predictions[-1]) == 0:
            return
        prediction = self.predictions[-1][0]
        sample_token = prediction.sample
        inst_token = prediction.instance
        output_trajs = self.pipeline.predictor.output_trajs
        output_probs = self.pipeline.predictor.output_probs
        res = 0.1/4
        track_helper = TrackingResultsPredictHelper(self.nuscenes, self.tracks[-1]['results'])
        agent_rasterizer = AgentBoxesFromTrackingPost(track_helper, resolution=res, seconds_of_history=self.pipeline.predictor.seconds_of_history)
        map_rasterizer = StaticLayerFromTrackingBW(track_helper, resolution=res)
        input_representation = input_representation_tracks(map_rasterizer, agent_rasterizer, Rasterizer())
        nusc_helper = PredictHelper(self.nuscenes)

        gt_future = self.sample_to_gt_pred[sample_token]
        predicted_track = self.pipeline.predictor.predicted_track
        gt_future_local = convert_global_coords_to_local(gt_future, predicted_track['translation'], predicted_track['rotation'])

        stacked_ground_truth = stack_ground_truth(gt_future_local, prediction.number_of_modes)
        mean_dists = mean_distances(np.array(output_trajs), stacked_ground_truth).flatten()
        dist_sorted_idxs = np.argsort(mean_dists)

        ## Get input image
        nearest_track = _find_nearest_track(track_helper, inst_token, sample_token)
        if nearest_track == None:
            return
        track_instance = nearest_track['tracking_id']

        input_image = input_representation.make_input_representation(track_instance, sample_token)

        render_path = render_path = '/mnt/hdd/output/ford/ast/fog/plots/'
        savepath = render_path + str(self.step) + '_' + inst_token + '_' + sample_token

        cmap = matplotlib.cm.get_cmap('tab10')
        msize=7
        lwidth = 1.4
        fig, ax = plt.subplots(1, 1, figsize=(9, 9))
        plt.imshow(input_image)
        for i in range(5):
            traj = output_trajs[i]
            label = 'p={:4.3f}'.format(output_probs[i])
            plt.plot(10.*traj[:, 0] + 250, -(10.*traj[:, 1]) + 400, color=cmap(i), marker='o', markersize=msize, label=label, zorder=10-i)

        plt.plot(10.*gt_future_local[:, 0] + 250, -(10.*gt_future_local[:, 1]) + 400, color='black', marker='o', markersize=msize, linewidth=lwidth, label='GT')
        leg = plt.legend(frameon=True, facecolor='white', framealpha=1, prop={'size': 14})
        plt.xlim(0, 500)
        plt.ylim(500, 0)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
                
        plt.savefig(savepath, bbox_inches='tight')
        plt.close('all')
        return ax



class OpenLoopPerceptionSimulator:
    def __init__(self,
                 nuscenes: NuScenes,
                 pipeline: ClassicalPerceptionPipeline,
                 pipeline_config: Dict = {},
                 scene_token: str = 'fcbccedd61424f1b85dcbf8f897f9754',
                 target_instance: str = '045cd82a77a1472499e8c15100cb5ff3',
                 eval_samples: List = [],
                 scatter_fraction = 0.05, # [-]
                 fog_density = 0.005, # 0.0 - 0.08 [1/m]
                 max_range = 25,
                 eval_metric = 'MinADEK',
                 eval_k = 10,
                 eval_metric_th=5.0,
                 no_pred_failure = False) -> None:

        assert eval_metric in ['MinADEK', 'MinFDEK']
        assert eval_k in [1, 5, 10]

        self.nuscenes = nuscenes
        self.pipeline = pipeline
        self.pipeline_config = pipeline_config
        # self.drop_likelihood = drop_likelihood
        self.no_pred_failure = no_pred_failure # Fail if no prediction during an eval sample?
        self._action = None
        self.target_instance = target_instance
        self.predict_eval_config_name = 'predict_2020_icra.json'

        self.beta = BetaRadomization(fog_density, 0)
        self.scatter_fraction = scatter_fraction
        self.fog_density = fog_density

        # Get all samples for scene
        ordered_samples = get_ordered_samples(nuscenes, scene_token)

        # find samples where instance is active
        target_ann_tokens = set(nuscenes.field2token('sample_annotation', 'instance_token', target_instance))
        target_annotations = [nuscenes.get('sample_annotation', t) for t in target_ann_tokens]
        target_annotations = sorted(target_annotations, key=lambda x: ordered_samples.index(x['sample_token']))
        
        last_target_sample_token = target_annotations[-1]['sample_token']
        ordered_samples = ordered_samples[:ordered_samples.index(last_target_sample_token)]
        self.ordered_samples = ordered_samples[:len(ordered_samples)-12]
        
        target_annotations = [ann for ann in target_annotations if ann['sample_token'] in self.ordered_samples]
    
        # filter samples for when target instance is within in ~20 m
        filtered_target_annotations = []
        for idx, sample_ann in enumerate(target_annotations):
            sample_token = sample_ann['sample_token']
            sample_rec = nuscenes.get('sample', sample_token)
            sd_record = nuscenes.get('sample_data', sample_rec['data']['LIDAR_TOP'])
            pose_record = nuscenes.get('ego_pose', sd_record['ego_pose_token'])


            # Both boxes and ego pose are given in global coord system, so distance can be calculated directly.
            box = nuscenes.get_box(sample_ann['token'])
            ego_translation = (box.center[0] - pose_record['translation'][0],
                               box.center[1] - pose_record['translation'][1],
                               box.center[2] - pose_record['translation'][2])
            if np.linalg.norm(ego_translation) <= max_range:
                filtered_target_annotations.append(sample_ann)

        self.target_annotations = filtered_target_annotations
        self.target_samples = [ann['sample_token'] for ann in self.target_annotations]

        assert len(self.target_samples) > 2, \
               'At least 2 valid samples required, found {}'.format(len(self.target_samples))

        if len(eval_samples) == 0:
            self.eval_samples = self.target_samples
            self.eval_annotations = self.target_annotations
        else:
            self.eval_samples = eval_samples
            self.eval_annotations = [ann for ann in self.target_annotations if ann['sample_token'] in self.eval_samples]


        [print(self.ordered_samples.index(t)) for t in self.eval_samples]

        # Load all PCDet data for scene
        pcdet_dataset = pipeline.pcdet_dataset
        self.pcdet_infos = {}
        for data_dict in pcdet_dataset:
            if data_dict['metadata']['token'] in self.ordered_samples:
                self.pcdet_infos[data_dict['metadata']['token']] = data_dict

        # Get boxes for target instance
        #sample_to_target_boxes = {}
        self.sample_to_point_idxs = {}
        self.sample_to_boxes = {}
        self.sample_to_gt_pred = {}
        nusc_helper = PredictHelper(self.nuscenes)
        self.predict_eval_config = load_prediction_config(nusc_helper, self.predict_eval_config_name)
        for ann in self.eval_annotations:
            data_dict = self.pcdet_infos[ann['sample_token']]
            boxes = get_boxes_for_pcdet_data(self.nuscenes, data_dict, ann)
            self.sample_to_boxes[ann['sample_token']] = boxes
        
            points = data_dict['points'] # [n x 5]

            masks = [points_in_box(b, points[:, :3].T, wlh_factor=1) for b in boxes]
            overall_mask = np.logical_or.reduce(np.array(masks))
            idxs_in_box = overall_mask.nonzero()[0]

            self.sample_to_point_idxs[ann['sample_token']] = idxs_in_box

            # Run detection for samples outside eval_samples 
            base_data_dicts = [d for key,d in self.pcdet_infos.items() if key not in self.eval_samples]
            self.base_detections = self.pipeline.run_detection(base_data_dicts)

            # Pre-calculate ground truth trajectory predictions
            sample_future = nusc_helper.get_future_for_agent(self.target_instance,
                                                             ann['sample_token'],
                                                             seconds=6.0,
                                                             in_agent_frame=False)
            self.sample_to_gt_pred[ann['sample_token']] = sample_future

        # Get predict tokens
        self.predict_tokens = self.get_predict_tokens()

        self.s0 = 0
        self.eval_idx = self.s0
        self.horizon = len(self.eval_samples)
        self.step = 0
        self.action = None
        self.info = []

        self.eval_metric = eval_metric
        self.eval_k = eval_k
        self.eval_metric_th = eval_metric_th
        
        self.detections = []
        self.tracks = []
        self.predictions = []
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

    def step_simulation(self, action: int):
        idx = self.step
        self.action = action
        sample_token = self.eval_samples[idx]

        data_dict = self.pcdet_infos[sample_token]
        points = data_dict['points']
        new_points, action_prob = haze_point_cloud(points, self.beta, self.scatter_fraction, action)
        self.new_points = new_points
        new_data_dict = update_data_dict(self.pipeline.pcdet_dataset, data_dict, new_points[:, :5])
        self.action_prob = action_prob
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
        for sample in self.base_detections['results'].keys():
            sample_det = {sample: self.base_detections['results'][sample]}
            self.pipeline.run_tracking(sample_det, batch_mode=False)

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

    def render(self):
        det_ax = self._render_detection()
        pred_ax = self._render_prediction()
        #self._render_disturbance()
        return det_ax, pred_ax
        
    def _render_detection(self):
        if len(self.detections) == 0:
            return

        render_path = '/mnt/hdd/output/ford/ast/fog/plots/'
        det_boxes = self.detections[-1]
        sample_token = det_boxes.sample_tokens[0]
        gt_boxes = load_sample_gt(self.nuscenes, sample_token, DetectionBox)

        savepath = render_path + str(self.step) + '_' + sample_token
        points = self.new_points[:, :5]

        ax = vis_sample_pointcloud(self.nuscenes,
                                   sample_token,
                                   gt_boxes=gt_boxes,
                                   pred_boxes=det_boxes,
                                   # pc=self.pcdet_infos[sample]['points'][self.sample_to_point_idxs[sample], :].T,
                                   pc=points.T,
                                   savepath=savepath)
        plt.sca(ax)


        render_path = render_path = '/mnt/hdd/output/ford/ast/fog/plots/'
        savepath = render_path + str(self.step) + '_'+ sample_token
        plt.savefig(savepath, bbox_inches='tight')

        return ax

    def _render_prediction(self):
        import matplotlib.cm
        from nuscenes.eval.prediction.metrics import stack_ground_truth, mean_distances

        if len(self.predictions[-1]) == 0:
            return

        prediction = self.predictions[-1][0]
        sample_token = prediction.sample
        inst_token = prediction.instance
        input_image = self.pipeline.predictor.input_image
        output_trajs = self.pipeline.predictor.output_trajs
        output_probs = self.pipeline.predictor.output_probs

        gt_future = self.sample_to_gt_pred[sample_token]
        predicted_track = self.pipeline.predictor.predicted_track
        gt_future_local = convert_global_coords_to_local(gt_future, predicted_track['translation'], predicted_track['rotation'])

        stacked_ground_truth = stack_ground_truth(gt_future_local, prediction.number_of_modes)
        mean_dists = mean_distances(np.array(output_trajs), stacked_ground_truth).flatten()
        dist_sorted_idxs = np.argsort(mean_dists)

        render_path = render_path = '/mnt/hdd/output/ford/ast/fog/plots/'
        savepath = render_path + str(self.step) + '_' + inst_token + '_' + sample_token

        cmap = matplotlib.cm.get_cmap('tab10')
        msize=7
        lwidth = 2.0
        fig, ax = plt.subplots(1, 1, figsize=(9, 9))
        plt.imshow(input_image)
        #for traj in output_trajs:
        for i in range(5):
            traj = output_trajs[i]
            label = 'p={:4.3f}'.format(output_probs[i])
            plt.plot(10.*traj[:, 0] + 250, -(10.*traj[:, 1]) + 400, color=cmap(i), marker='o', markersize=msize, label=label, zorder=10-i)

        plt.plot(10.*gt_future_local[:, 0] + 250, -(10.*gt_future_local[:, 1]) + 400, color='black', marker='o', markersize=msize, linewidth=lwidth, label='GT')
        leg = plt.legend(frameon=True, facecolor='white', framealpha=1, prop={'size': 14})
        plt.xlim(0, 500)
        plt.ylim(500, 0)
        # Hide grid lines
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.savefig(savepath, bbox_inches='tight')
        plt.close('all')
        return ax
