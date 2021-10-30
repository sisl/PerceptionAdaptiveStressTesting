from typing import Dict, List
import pickle
import numpy as np
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
import torch
from torch.nn.functional import softmax

from nuscenes.prediction.helper import  convert_local_coords_to_global

from nuscenes.prediction import PredictHelper
from nuscenes.prediction.input_representation.interface import InputRepresentation
from nuscenes.prediction.input_representation.combinators import Rasterizer

from nuscenes.prediction.models.backbone import ResNetBackbone
from nuscenes.prediction.models.covernet import CoverNet

from sequential_perception.input_representation_tracks import AgentBoxesFromTracking, StaticLayerFromTracking
from sequential_perception.predict_helper_tracks import TrackingResultsPredictHelper

from nuscenes.eval.prediction.data_classes import Prediction


class CoverNetTrackPredictor:
    def __init__(self, track_helper: TrackingResultsPredictHelper,
                 path_to_traj_sets: str,
                 track_gt_match_thresh: float = 2.0,
                 seconds_of_history: float = 1.0,
                 num_output_modes: int = 10,
                 num_modes: int = 64,
                 path_to_weights: str = None,
                 use_cuda=False,
                 plot_path:str = None
                 ):
        self.track_helper = track_helper
        self.nusc_helper = PredictHelper(track_helper.data)
        self.num_output_modes = num_output_modes
        self.use_cuda = False
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.num_timesteps = 12
        self.plot_path = plot_path
        self.track_gt_match_threshold = track_gt_match_thresh


        # CoverNet
        backbone = ResNetBackbone('resnet50')
        self.covernet = CoverNet(backbone, num_modes=num_modes).to(self.device)
        if path_to_weights != None:
            #self.covernet.load_state_dict(torch.jit.load(path_to_weights))
            self.covernet.load_state_dict(torch.load(path_to_weights))
            #print('wanted weights ...')
        self.covernet.eval()

        # Trajectories
        trajectories = pickle.load(open(path_to_traj_sets, 'rb'))
        self.trajectories = torch.Tensor(trajectories).to(self.device)

        # Input Representation
        agent_rasterizer = AgentBoxesFromTracking(track_helper, seconds_of_history=seconds_of_history)
        map_rasterizer = StaticLayerFromTracking(track_helper)
        self.input_representation = InputRepresentation(map_rasterizer, agent_rasterizer, Rasterizer())

    def _get_agent_state(self, instance: str, sample: str):
        """
        Returns a vector of velocity magnitude, acceleration magnitude, and yaw rate from
        the given instance (track ID) and sample.
        :instance: Token of instance.
        :sample: Token of sample.
        :return: 
        """
        helper = self.track_helper

        velocity = helper.get_velocity_for_agent(instance, sample)
        acceleration = helper.get_acceleration_for_agent(instance, sample)
        yaw_rate = helper.get_heading_change_rate_for_agent(instance, sample)

        if np.isnan(velocity):
            velocity = 0.0
        if np.isnan(acceleration):
            acceleration = 0.0
        if np.isnan(yaw_rate):
            yaw_rate = 0.0

        return np.array([velocity, acceleration, yaw_rate], dtype=float)

    def _render_prediction(self, token: str, input_image: np.array, trajectories: List) -> None:
        plt.imshow(input_image)
        for traj in trajectories:
        #for i in range():
            #most_likely_traj = trajectories[logits[0].argsort(descending=True)[i]].cpu() # :5
            plt.scatter(10.*traj[:,0]+250, -(10.*traj[:,1])+400, color='green')
            plt.xlim(0,500)
            plt.ylim(500,0)
        plt.savefig(self.plot_path + '/' + token +'.png')
        plt.close('all')
        return

    def _find_nearest_track(self, gt_instance, sample):
        gt_annotation = self.nusc_helper.get_sample_annotation(gt_instance, sample)
        gt_pos = np.array(gt_annotation['translation'][:2])

        #tracks = tracking_results[sample_token]
        tracks = self.track_helper.get_annotations_for_sample(sample)
        min_dist = np.inf
        best_track = None
        for track in tracks:
            if track['tracking_name'] in ['car', 'bus', 'truck']:
                track_pos = np.array(track['translation'][:2])
                dist = np.linalg.norm(track_pos - gt_pos)
                if dist < min_dist:
                    min_dist = dist
                    best_track = track

        if min_dist > self.track_gt_match_threshold:
            print('Track could not be matched for instance={} and sample={}'.format(gt_instance, sample))
            best_track = None

        return best_track


    def __call__(self, token: str) -> Prediction:
        """
        Makes prediction.
        :param token: string of format {instance_token}_{sample_token}.
        """
        gt_instance, sample = token.split("_")

        # Find nearest track
        nearest_track = self._find_nearest_track(gt_instance, sample)
        if nearest_track == None:
            return None
        
        instance = nearest_track['tracking_id']
        
        agent_state_vector = self._get_agent_state(instance, sample)
        agent_img = self.input_representation.make_input_representation(instance, sample)


        # agent_state_vector = torch.Tensor([[pred_helper.get_velocity_for_agent(gt_instance_token, sample_token),
        #                                     pred_helper.get_acceleration_for_agent(gt_instance_token, sample_token),
        #                                     pred_helper.get_heading_change_rate_for_agent(gt_instance_token, sample_token)]])
        agent_state_tensor= torch.Tensor([agent_state_vector]).to(self.device)
        image_tensor = torch.Tensor(agent_img).permute(2, 0, 1).unsqueeze(0).to(self.device)

        logits = self.covernet(image_tensor, agent_state_tensor)

        #predicted_trajs = []
        #for i in range(self.num_output_modes):
        most_likely_trajs = self.trajectories[logits[0].argsort(descending=True)[:self.num_output_modes]].cpu() # :5
        probabilities = softmax(logits[0].sort(descending=True).values[:self.num_output_modes], dim=0).cpu()

        # Convert trajectories from local to global frame
        center_agent_annotation = self.track_helper.get_sample_annotation(instance, sample)
        center_translation = center_agent_annotation['translation'][:2]
        center_rotation = center_agent_annotation['rotation'] 
        
        prediction_global_trajectories = np.zeros((self.num_output_modes, self.num_timesteps, 2))
        for j,traj in enumerate(most_likely_trajs):
            # traj_global = np.zeros(traj.shape)
            # for i in range(traj.shape[0]):
            #     traj_global[i, :] = convert_local_coords_to_global(traj[i, :], center_translation, center_rotation)
            traj_global =  convert_local_coords_to_global(traj, center_translation, center_rotation)
            prediction_global_trajectories[j, :, :] = traj_global

        if self.plot_path != None:
            self._render_prediction(token, agent_img, most_likely_trajs)

        return Prediction(gt_instance, sample, prediction_global_trajectories, probabilities.detach().numpy())


class PipelineCoverNetModule:
    def __init__(self,
                 nuscenes : NuScenes,
                 eps_sets_path: str,
                 weights_path: str,
                 track_gt_match_thresh: float = 2.0,
                 seconds_of_history: float = 1.0,
                 num_output_modes: int = 10,
                 num_modes: int = 64,
                 use_cuda=True,
                 plot_path:str = None
                 ):
        #self.track_helper = track_helper
        # self.nusc_helper = PredictHelper(track_helper.data)
        self.nuscenes = nuscenes
        self.nusc_helper = PredictHelper(nuscenes)
        self.num_output_modes = num_output_modes
        self.use_cuda = False
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.num_timesteps = 12
        self.plot_path = plot_path
        self.track_gt_match_threshold = track_gt_match_thresh
        self.seconds_of_history = seconds_of_history
        self.input_image = None
        self.predicted_track = None


        # CoverNet
        backbone = ResNetBackbone('resnet50')
        self.covernet = CoverNet(backbone, num_modes=num_modes)
        self.covernet.load_state_dict(torch.load(weights_path))
        self.covernet.cuda()
        self.covernet.eval()

        # Trajectories
        trajectories = pickle.load(open(eps_sets_path, 'rb'))
        self.trajectories = torch.Tensor(trajectories).to(self.device)

    def _get_agent_state(self, track_helper, instance: str, sample: str):
        """
        Returns a vector of velocity magnitude, acceleration magnitude, and yaw rate from
        the given instance (track ID) and sample.
        :instance: Token of instance.
        :sample: Token of sample.
        :return: 
        """
        helper = track_helper

        velocity = helper.get_velocity_for_agent(instance, sample)
        acceleration = helper.get_acceleration_for_agent(instance, sample)
        yaw_rate = helper.get_heading_change_rate_for_agent(instance, sample)

        if np.isnan(velocity):
            velocity = 0.0
        if np.isnan(acceleration):
            acceleration = 0.0
        if np.isnan(yaw_rate):
            yaw_rate = 0.0

        return np.array([velocity, acceleration, yaw_rate], dtype=float)

    def _render_prediction(self, token: str, input_image: np.ndarray, trajectories: List) -> None:
        plt.imshow(input_image)
        for traj in trajectories:
        #for i in range():
            #most_likely_traj = trajectories[logits[0].argsort(descending=True)[i]].cpu() # :5
            plt.scatter(10.*traj[:, 0] + 250, -(10.*traj[:, 1]) + 400, color='green')
            plt.xlim(0, 500)
            plt.ylim(500, 0)
        plt.savefig(self.plot_path + '/' + token +'.png')
        plt.close('all')
        return

    def _find_nearest_track(self, track_helper, gt_instance, sample):
        gt_annotation = self.nusc_helper.get_sample_annotation(gt_instance, sample)
        gt_pos = np.array(gt_annotation['translation'][:2])

        #tracks = tracking_results[sample_token]
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

        if min_dist > self.track_gt_match_threshold:
            print('Track could not be matched for instance={} and sample={}'.format(gt_instance, sample))
            best_track = None

        return best_track

    @torch.no_grad()
    def __call__(self, tokens: List[str], tracks: dict) -> List[Prediction]:
        """
        Makes prediction.
        :param token: string of format {instance_token}_{sample_token}.
        """
        track_helper = TrackingResultsPredictHelper(self.nuscenes, tracks['results'])
        agent_rasterizer = AgentBoxesFromTracking(track_helper, seconds_of_history=self.seconds_of_history)
        map_rasterizer = StaticLayerFromTracking(track_helper)
        input_representation = InputRepresentation(map_rasterizer, agent_rasterizer, Rasterizer())
        
        predictions = []
        for token in tokens:
            gt_instance, sample = token.split("_")

            # Find nearest track
            nearest_track = self._find_nearest_track(track_helper, gt_instance, sample)
            if nearest_track == None:
                self.input_image = None
                self.output_trajs = None
                self.output_probs = None
                self.predicted_track = None
                return None
            self.predicted_track = nearest_track
            
            instance = nearest_track['tracking_id']
            
            agent_state_vector = self._get_agent_state(track_helper, instance, sample)
            agent_img = input_representation.make_input_representation(instance, sample)
            self.input_image = agent_img

            agent_state_tensor= torch.Tensor([agent_state_vector]).to(self.device)
            image_tensor = torch.Tensor(agent_img).to(self.device).permute(2, 0, 1).unsqueeze(0)

            logits = self.covernet(image_tensor, agent_state_tensor)

            most_likely_trajs = self.trajectories[logits[0].argsort(descending=True)[:self.num_output_modes]].cpu() # :5
            probabilities = softmax(logits[0].sort(descending=True).values[:self.num_output_modes], dim=0).cpu()
            self.output_trajs = most_likely_trajs.numpy()
            self.output_probs = probabilities.numpy()

            # Convert trajectories from local to global frame
            center_agent_annotation = track_helper.get_sample_annotation(instance, sample)
            center_translation = center_agent_annotation['translation'][:2]
            center_rotation = center_agent_annotation['rotation'] 
            
            prediction_global_trajectories = np.zeros((self.num_output_modes, self.num_timesteps, 2))
            for j,traj in enumerate(most_likely_trajs):
                traj_global = convert_local_coords_to_global(traj, center_translation, center_rotation)
                prediction_global_trajectories[j, :, :] = traj_global

            if self.plot_path != None:
                self._render_prediction(token, agent_img, most_likely_trajs)

            prediction = Prediction(gt_instance, sample, prediction_global_trajectories, probabilities.detach().numpy())
            predictions.append(prediction)

        return predictions



class CoverNetPredictModule:
    def __init__(self,
                 covernet,
                 trajectory_sets,
                 nuscenes : NuScenes,
                 track_gt_match_thresh: float = 2.0,
                 seconds_of_history: float = 1.0,
                 num_output_modes: int = 10,
                #  num_modes: int = 64,
                 use_cuda=True,
                 plot_path:str = None
                 ):
        self.nuscenes = nuscenes
        self.nusc_helper = PredictHelper(nuscenes)
        self.num_output_modes = num_output_modes
        self.use_cuda = False
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.num_timesteps = 12
        self.plot_path = plot_path
        self.track_gt_match_threshold = track_gt_match_thresh
        self.seconds_of_history = seconds_of_history
        self.input_image = None
        self.predicted_track = None


        # CoverNet
        # backbone = ResNetBackbone('resnet50')
        # self.covernet = CoverNet(backbone, num_modes=num_modes)
        # self.covernet.load_state_dict(torch.load(weights_path))
        # self.covernet.cuda()
        # self.covernet.eval()
        self.covernet = covernet

        # Trajectories
        # trajectories = pickle.load(open(eps_sets_path, 'rb'))
        # self.trajectories = torch.Tensor(trajectories).to(self.device)
        self.trajectories = trajectory_sets

    def _get_agent_state(self, track_helper, instance: str, sample: str):
        """
        Returns a vector of velocity magnitude, acceleration magnitude, and yaw rate from
        the given instance (track ID) and sample.
        :instance: Token of instance.
        :sample: Token of sample.
        :return: 
        """
        helper = track_helper

        velocity = helper.get_velocity_for_agent(instance, sample)
        acceleration = helper.get_acceleration_for_agent(instance, sample)
        yaw_rate = helper.get_heading_change_rate_for_agent(instance, sample)

        if np.isnan(velocity):
            velocity = 0.0
        if np.isnan(acceleration):
            acceleration = 0.0
        if np.isnan(yaw_rate):
            yaw_rate = 0.0

        return np.array([velocity, acceleration, yaw_rate], dtype=float)

    def _render_prediction(self, token: str, input_image: np.ndarray, trajectories: List) -> None:
        plt.imshow(input_image)
        for traj in trajectories:
        #for i in range():
            #most_likely_traj = trajectories[logits[0].argsort(descending=True)[i]].cpu() # :5
            plt.scatter(10.*traj[:, 0] + 250, -(10.*traj[:, 1]) + 400, color='green')
            plt.xlim(0, 500)
            plt.ylim(500, 0)
        plt.savefig(self.plot_path + '/' + token +'.png')
        plt.close('all')
        return

    def _find_nearest_track(self, track_helper, gt_instance, sample):
        gt_annotation = self.nusc_helper.get_sample_annotation(gt_instance, sample)
        gt_pos = np.array(gt_annotation['translation'][:2])

        #tracks = tracking_results[sample_token]
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

        if min_dist > self.track_gt_match_threshold:
            print('Track could not be matched for instance={} and sample={}'.format(gt_instance, sample))
            best_track = None

        return best_track

    @torch.no_grad()
    def __call__(self, tokens: List[str], tracks: dict) -> List[Prediction]:
        """
        Makes prediction.
        :param token: string of format {instance_token}_{sample_token}.
        """
        track_helper = TrackingResultsPredictHelper(self.nuscenes, tracks['results'])
        agent_rasterizer = AgentBoxesFromTracking(track_helper, seconds_of_history=self.seconds_of_history)
        map_rasterizer = StaticLayerFromTracking(track_helper)
        input_representation = InputRepresentation(map_rasterizer, agent_rasterizer, Rasterizer())
        
        predictions = []
        for token in tokens:
            gt_instance, sample = token.split("_")

            # Find nearest track
            nearest_track = self._find_nearest_track(track_helper, gt_instance, sample)
            if nearest_track == None:
                self.input_image = None
                self.output_trajs = None
                self.output_probs = None
                self.predicted_track = None
                return None
            self.predicted_track = nearest_track
            
            instance = nearest_track['tracking_id']
            
            agent_state_vector = self._get_agent_state(track_helper, instance, sample)
            agent_img = input_representation.make_input_representation(instance, sample)
            self.input_image = agent_img

            agent_state_tensor= torch.Tensor([agent_state_vector]).to(self.device)
            image_tensor = torch.Tensor(agent_img).to(self.device).permute(2, 0, 1).unsqueeze(0)

            logits = self.covernet(image_tensor, agent_state_tensor)

            most_likely_trajs = self.trajectories[logits[0].argsort(descending=True)[:self.num_output_modes]].cpu() # :5
            probabilities = softmax(logits[0].sort(descending=True).values[:self.num_output_modes], dim=0).cpu()
            self.output_trajs = most_likely_trajs.numpy()
            self.output_probs = probabilities.numpy()

            # Convert trajectories from local to global frame
            center_agent_annotation = track_helper.get_sample_annotation(instance, sample)
            center_translation = center_agent_annotation['translation'][:2]
            center_rotation = center_agent_annotation['rotation'] 
            
            prediction_global_trajectories = np.zeros((self.num_output_modes, self.num_timesteps, 2))
            for j,traj in enumerate(most_likely_trajs):
                traj_global = convert_local_coords_to_global(traj, center_translation, center_rotation)
                prediction_global_trajectories[j, :, :] = traj_global

            if self.plot_path != None:
                self._render_prediction(token, agent_img, most_likely_trajs)

            prediction = Prediction(gt_instance, sample, prediction_global_trajectories, probabilities.detach().numpy())
            predictions.append(prediction)

        return predictions



def find_closest_track_to_instance(nusc_helper: PredictHelper,
                                    tracking_results: Dict,
                                    sample_token: str,
                                    instance_token: str):
    """
    Returns the closest track to a given instance from NuScenes
    :nusc_helper: Nuscenes PredictHelper
    :tracking_results: Results from tracking
    :sample_token: Token of sample.
    :instance_token: Token of instance.
    :return: 
    """

    gt_annotation = nusc_helper.get_sample_annotation(instance_token, sample_token)
    gt_pos = np.array(gt_annotation['translation'][:2])


    tracks = tracking_results[sample_token]
    min_dist = np.inf
    best_track = None
    for track in tracks:
        if track['tracking_name'] in ['car', 'truck', 'bus']:
            track_pos = np.array(track['translation'][:2])
            dist = np.linalg.norm(track_pos - gt_pos)
            if dist < min_dist:
                min_dist = dist
                best_track = track
        if min_dist <= 2.0:
            break

    return best_track