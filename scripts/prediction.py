from os import path
from typing import Any, Dict, List, Tuple, Callable, Union
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from nuscenes import prediction
import torch
from torch.nn.functional import softmax
import pickle
from nuscenes.prediction.helper import BUFFER, Record, angle_of_rotation, convert_global_coords_to_local
from nuscenes.eval.common.loaders import load_prediction
#from nuscenes.eval.common.config import config_factory
from pyquaternion import Quaternion
from nuscenes.prediction.helper import quaternion_yaw, convert_local_coords_to_global
#from sequential_perception.tracking_render import CustomTrackingRenderer
from nuscenes.eval.tracking.data_classes import TrackingBox
from nuscenes.eval.tracking.loaders import create_tracks
from nuscenes.prediction import PredictHelper
from nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistory #, draw_agent_boxes
from nuscenes import NuScenes
from nuscenes.eval.prediction.splits import get_prediction_challenge_split

from nuscenes.prediction.input_representation.static_layers import Color, StaticLayerRasterizer
from nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistory
from nuscenes.prediction.input_representation.interface import InputRepresentation
from nuscenes.prediction.input_representation.combinators import Rasterizer

from nuscenes.prediction.input_representation.interface import StaticLayerRepresentation
from nuscenes.prediction.input_representation.static_layers import load_all_maps

from nuscenes.prediction.models.backbone import ResNetBackbone
# from nuscenes.prediction.models.mtp import MTP
from nuscenes.prediction.models.covernet import CoverNet

# from sequential_perception.covernet_module import CoverNetModel
from sequential_perception.input_representation_tracks import AgentBoxesFromTracking, StaticLayerFromTracking
from sequential_perception.predict_helper_tracks import TrackingResultsPredictHelper

from nuscenes.eval.prediction.data_classes import Prediction

from sequential_perception.predictors import CoverNetTrackPredictor

#Record = Dict[str, Any]
#History = Dict[str, List[Dict[str, Any]]]
#BUFFER = 0.15  # seconds

# class CoverNetTrackPredictor:
#     def __init__(self, track_helper: TrackingResultsPredictHelper,
#                  path_to_traj_sets: str,
#                  track_gt_match_thresh: float = 2.0,
#                  seconds_of_history: float = 2.0,
#                  num_output_modes: int = 5,
#                  num_modes: int = 64,
#                  path_to_weights: str = None,
#                  use_cuda=False,
#                  plot_path:str = None
#                  ):
#         self.track_helper = track_helper
#         self.nusc_helper = PredictHelper(track_helper.data)
#         self.num_output_modes = num_output_modes
#         self.use_cuda = False
#         self.device = torch.device("cuda" if use_cuda else "cpu")
#         self.num_timesteps = 12
#         self.plot_path = plot_path
#         self.track_gt_match_threshold = track_gt_match_thresh


#         # CoverNet
#         backbone = ResNetBackbone('resnet50')
#         self.covernet = CoverNet(backbone, num_modes=num_modes).to(self.device)
#         if path_to_weights != None:
#             #self.covernet.load_state_dict(torch.jit.load(path_to_weights))
#             self.covernet.load_state_dict(torch.load(path_to_weights))
#             #print('wanted weights ...')
#         self.covernet.eval()

#         # Trajectories
#         trajectories = pickle.load(open(path_to_traj_sets, 'rb'))
#         self.trajectories = torch.Tensor(trajectories).to(self.device)

#         # Input Representation
#         agent_rasterizer = AgentBoxesFromTracking(track_helper, seconds_of_history=seconds_of_history)
#         map_rasterizer = StaticLayerFromTracking(track_helper)
#         self.input_representation = InputRepresentation(map_rasterizer, agent_rasterizer, Rasterizer())

#     def _get_agent_state(self, instance: str, sample: str):
#         """
#         Returns a vector of velocity magnitude, acceleration magnitude, and yaw rate from
#         the given instance (track ID) and sample.
#         :instance: Token of instance.
#         :sample: Token of sample.
#         :return: 
#         """
#         helper = self.track_helper

#         velocity = helper.get_velocity_for_agent(instance, sample)
#         acceleration = helper.get_acceleration_for_agent(instance, sample)
#         yaw_rate = helper.get_heading_change_rate_for_agent(instance, sample)

#         if np.isnan(velocity):
#             velocity = 0.0
#         if np.isnan(acceleration):
#             acceleration = 0.0
#         if np.isnan(yaw_rate):
#             yaw_rate = 0.0

#         return np.array([velocity, acceleration, yaw_rate], dtype=float)

#     def _render_prediction(self, token: str, input_image: np.array, trajectories: List) -> None:
#         plt.imshow(input_image)
#         for traj in trajectories:
#         #for i in range():
#             #most_likely_traj = trajectories[logits[0].argsort(descending=True)[i]].cpu() # :5
#             plt.scatter(10.*traj[:,0]+250, -(10.*traj[:,1])+400, color='green')
#             plt.xlim(0,500)
#             plt.ylim(500,0)
#         plt.savefig(self.plot_path + '/' + token +'.png')
#         plt.close()
#         return

#     def _find_nearest_track(self, gt_instance, sample):
#         gt_annotation = self.nusc_helper.get_sample_annotation(gt_instance, sample)
#         gt_pos = np.array(gt_annotation['translation'][:2])

#         #tracks = tracking_results[sample_token]
#         tracks = self.track_helper.get_annotations_for_sample(sample)
#         min_dist = np.inf
#         best_track = None
#         for track in tracks:
#             if track['tracking_name'] in ['car', 'bus', 'truck']:
#                 track_pos = np.array(track['translation'][:2])
#                 dist = np.linalg.norm(track_pos - gt_pos)
#                 if dist < min_dist:
#                     min_dist = dist
#                     best_track = track

#         if min_dist > self.track_gt_match_threshold:
#             print('Track could not be matched for instance={} and sample={}'.format(gt_instance, sample))
#             best_track = None

#         return best_track


#     def __call__(self, token: str) -> Prediction:
#         """
#         Makes prediction.
#         :param token: string of format {instance_token}_{sample_token}.
#         """
#         gt_instance, sample = token.split("_")

#         # Find nearest track
#         nearest_track = self._find_nearest_track(gt_instance, sample)
#         if nearest_track == None:
#             return None
        
#         instance = nearest_track['tracking_id']
        
#         agent_state_vector = self._get_agent_state(instance, sample)
#         agent_img = self.input_representation.make_input_representation(instance, sample)


#         # agent_state_vector = torch.Tensor([[pred_helper.get_velocity_for_agent(gt_instance_token, sample_token),
#         #                                     pred_helper.get_acceleration_for_agent(gt_instance_token, sample_token),
#         #                                     pred_helper.get_heading_change_rate_for_agent(gt_instance_token, sample_token)]])
#         agent_state_tensor= torch.Tensor([agent_state_vector]).to(self.device)
#         image_tensor = torch.Tensor(agent_img).permute(2, 0, 1).unsqueeze(0).to(self.device)

#         logits = self.covernet(image_tensor, agent_state_tensor)

#         #predicted_trajs = []
#         #for i in range(self.num_output_modes):
#         most_likely_trajs = self.trajectories[logits[0].argsort(descending=True)[:self.num_output_modes]].cpu() # :5
#         probabilities = softmax(logits[0].sort(descending=True).values[:self.num_output_modes], dim=0).cpu()

#         # Convert trajectories from local to global frame
#         center_agent_annotation = self.track_helper.get_sample_annotation(instance, sample)
#         center_translation = center_agent_annotation['translation'][:2]
#         center_rotation = center_agent_annotation['rotation'] 
        
#         prediction_global_trajectories = np.zeros((self.num_output_modes, self.num_timesteps, 2))
#         for j,traj in enumerate(most_likely_trajs):
#             # traj_global = np.zeros(traj.shape)
#             # for i in range(traj.shape[0]):
#             #     traj_global[i, :] = convert_local_coords_to_global(traj[i, :], center_translation, center_rotation)
#             traj_global =  convert_local_coords_to_global(traj, center_translation, center_rotation)
#             prediction_global_trajectories[j, :, :] = traj_global

#         if self.plot_path != None:
#             self._render_prediction(token, agent_img, most_likely_trajs)

#         return Prediction(gt_instance, sample, prediction_global_trajectories, probabilities.detach().numpy())


# def find_closest_track_to_instance(nusc_helper: PredictHelper,
#                                     tracking_results: Dict,
#                                     sample_token: str,
#                                     instance_token: str):

#     gt_annotation = nusc_helper.get_sample_annotation(instance_token, sample_token)
#     gt_pos = np.array(gt_annotation['translation'][:2])


#     tracks = tracking_results[sample_token]
#     min_dist = np.inf
#     best_track = None
#     for track in tracks:
#         track_pos = np.array(track['translation'][:2])
#         dist = np.linalg.norm(track_pos - gt_pos)
#         if dist < min_dist:
#             min_dist = dist
#             best_track = track

#     return best_track



# def main():
#     nusc_dataroot = '/scratch/hdelecki/ford/data/sets/nuscenes/v1.0-mini'
#     nusc_version = 'v1.0-mini'
#     split = 'mini_val'
    
#     tracking_results_path = '/scratch/hdelecki/ford/output/tracking/kalman/nusc_tracking_results.json'

#     img_output_path = '/scratch/hdelecki/ford/output/prediction/test/'

#     eps_set_path = '/scratch/hdelecki/ford/models/nuscenes-prediction-challenge-trajectory-sets/epsilon_8.pkl'
#     ckpt_path = '/scratch/hdelecki/ford/models/covernet_ckpts/epoch=24-step=50299.ckpt'

#     seconds_of_history = 2.0

#     # Get tracking results
#     #instance_token_img, sample_token_img = 'bc38961ca0ac4b14ab90e547ba79fbb6', '7626dde27d604ac28a0240bdd54eba7a'
#     all_results, meta = load_prediction(tracking_results_path, 1000, TrackingBox, verbose=True)
#     results_dict = all_results.serialize()

#     # Nuscenes and Helpers
#     nusc = NuScenes('v1.0-mini', dataroot=nusc_dataroot)
#     pred_helper = PredictHelper(nusc)
#     track_helper = TrackingResultsPredictHelper(nusc, results_dict)

#     # Covernet
#     backbone = ResNetBackbone('resnet50')
#     covernet = CoverNet(backbone, num_modes=64)


#     inst_sample_list = get_prediction_challenge_split(split, dataroot=nusc_dataroot)
#     #instance_token, sample_token = inst_sample_list[11].split("_")
#     gt_instance_token, sample_token = inst_sample_list[11].split("_")
#     #instance_token = '30055'
#     #instance_token = '30047'


#     # print(sample_token)
#     # print(instance_token)
#     assert sample_token in results_dict.keys()

#     # img = agent_rasterizer.make_representation(instance_token, sample_token)
#     # cv2.imwrite('./agent_raster_test.png', img[:, :, ::-1])

#     # agent_rasterizer = AgentBoxesWithFadedHistory(pred_helper)
#     # img = agent_rasterizer.make_representation(instance_token, sample_token)
#     # cv2.imwrite('./agent_raster_test.png', img[:, :, ::-1])

#     static_layer_rasterizer = StaticLayerRasterizer(pred_helper)
#     agent_rasterizer = AgentBoxesWithFadedHistory(pred_helper, seconds_of_history=seconds_of_history)
#     mtp_input_representation = InputRepresentation(static_layer_rasterizer, agent_rasterizer, Rasterizer())
#     img = mtp_input_representation.make_input_representation(gt_instance_token, sample_token)
#     cv2.imwrite('./true_full.png', img[:, :, ::-1])


#     agent_state_vector = torch.Tensor([[pred_helper.get_velocity_for_agent(gt_instance_token, sample_token),
#                                         pred_helper.get_acceleration_for_agent(gt_instance_token, sample_token),
#                                         pred_helper.get_heading_change_rate_for_agent(gt_instance_token, sample_token)]])
#     image_tensor = torch.Tensor(img).permute(2, 0, 1).unsqueeze(0)

#     # Assume NaN velocity/acceleration is zero
#     agent_state_vector[torch.isnan(agent_state_vector)] = 0.0


#     #print(agent_state_vector)
#     logits = covernet(image_tensor, agent_state_vector)
#     print(logits)


#     # Trajectories
#     trajectories = pickle.load(open(eps_set_path, 'rb'))
#     trajectories = torch.Tensor(trajectories)

#     plt.imshow(img)
#     for i in range(0,5):
#         most_likely_traj = trajectories[logits[0].argsort(descending=True)[i]].cpu() # :5

#         # Singapore or Boston for training? Both. Different sides of the road...
#         plt.scatter(10.*most_likely_traj[:,0]+250, -(10.*most_likely_traj[:,1])+400, color='green')
#         plt.xlim(0,500)
#         plt.ylim(500,0)
#     plt.savefig('./true_traj.png')





#     ####### Prediction from Tack ############

#     closest_track = find_closest_track_to_instance(pred_helper, results_dict, sample_token, gt_instance_token)
#     print(closest_track['tracking_id'])
#     #sample_token = closest_track['tracking_id']
#     instance_token = closest_track['tracking_id']

#     agent_rasterizer = AgentBoxesFromTracking(track_helper, seconds_of_history=seconds_of_history)
#     map_rasterizer = StaticLayerFromTracking(track_helper)

#     agent_img = agent_rasterizer.make_representation(instance_token, sample_token)
#     cv2.imwrite('./agent_raster_test_tracks.png', agent_img[:, :, ::-1])

#     map_img = map_rasterizer.make_representation(instance_token, sample_token)
#     cv2.imwrite('./map_raster_test_tracks.png', map_img[:, :, ::-1])

#     track_mtp_input_representation = InputRepresentation(map_rasterizer, agent_rasterizer, Rasterizer())
#     img = track_mtp_input_representation.make_input_representation(instance_token, sample_token)
#     cv2.imwrite('./full_track.png', img[:, :, ::-1])


#     vel = track_helper.get_velocity_for_agent(instance_token, sample_token)


#     agent_state_vector = torch.Tensor([[track_helper.get_velocity_for_agent(instance_token, sample_token),
#                                         track_helper.get_acceleration_for_agent(instance_token, sample_token),
#                                         track_helper.get_heading_change_rate_for_agent(instance_token, sample_token)]])
#     image_tensor = torch.Tensor(img).permute(2, 0, 1).unsqueeze(0)

#     # Assume NaN velocity/acceleration is zero
#     agent_state_vector[torch.isnan(agent_state_vector)] = 0


#     #print(agent_state_vector)
#     logits = covernet(image_tensor, agent_state_vector)
#     print(logits)

    
    
#     # model_test = CoverNetModel.load_from_checkpoint(ckpt_path).cuda()
#     # model_test.eval()
    
#     # logits = model_test.forward((image_tensor, agent_state_vector))
#     # print(logits)


#     #covernet = CoverNetModel(path_to_epsilon_set=eps_set_path)
#     #covernet.load_from_checkpoint(checkpoint_path=ckpt_path)


#     # Trajectories
#     trajectories = pickle.load(open(eps_set_path, 'rb'))
#     trajectories = torch.Tensor(trajectories)

#     plt.imshow(img)
#     for i in range(0,5):
#         most_likely_traj = trajectories[logits[0].argsort(descending=True)[i]].cpu() # :5

#         # Singapore or Boston for training? Both. Different sides of the road...
#         plt.scatter(10.*most_likely_traj[:,0]+250, -(10.*most_likely_traj[:,1])+400, color='green')
#         plt.xlim(0,500)
#         plt.ylim(500,0)
#     plt.savefig('./pred_traj.png')

#     return

def main():
    nusc_dataroot = '/scratch/hdelecki/ford/data/sets/nuscenes/v1.0-mini'
    nusc_version = 'v1.0-mini'
    split = 'mini_val'
    
    tracking_results_path = '/scratch/hdelecki/ford/output/tracking/kalman/nusc_tracking_results.json'

    eps_set_path = '/scratch/hdelecki/ford/models/nuscenes-prediction-challenge-trajectory-sets/epsilon_8.pkl'
    ckpt_path = '/scratch/hdelecki/ford/models/covernet_ckpts/epoch=24-step=50299.ckpt'
    #state_dict_path = '/scratch/hdelecki/ford/models/covernet_ckpts/epoch=24-step=50299-state-dict.pt'
    state_dict_path = '/scratch/hdelecki/ford/models/covernet_ckpts/epoch=24-step=50299-state-dict-torch13.pt'

    predction_results_path = '/scratch/hdelecki/ford/output/prediction/test_prediction'
    prediction_plot_path = '/scratch/hdelecki/ford/output/prediction/test_prediction/plots'
    #prediction_plot_path = None

    gate_distance = 2.0
    seconds_of_history = 1.0
    plot_gt = True
    num_output_modes = 10

    # Load NuScenes + Helper
    nusc = NuScenes(version=nusc_version, dataroot=nusc_dataroot)
    nusc_helper = PredictHelper(nusc)

    # Get tracking results
    all_results, meta = load_prediction(tracking_results_path, 1000, TrackingBox, verbose=True)
    tracking_results_dict = all_results.serialize()
    track_helper = TrackingResultsPredictHelper(nusc, tracking_results_dict)

    predictor = CoverNetTrackPredictor(track_helper=track_helper,
                                       path_to_traj_sets=eps_set_path,
                                       seconds_of_history=seconds_of_history,
                                       num_output_modes=num_output_modes,
                                       track_gt_match_thresh=gate_distance,
                                       plot_path=prediction_plot_path,
                                       path_to_weights=state_dict_path)

    prediction_challenge_tokens = get_prediction_challenge_split(split, dataroot=nusc_dataroot)
    print(len(prediction_challenge_tokens))

    if plot_gt:
        gt_static_layer = StaticLayerRasterizer(nusc_helper)
        gt_agent_rasterizer = AgentBoxesWithFadedHistory(nusc_helper, seconds_of_history=seconds_of_history)
        gt_representation = InputRepresentation(gt_static_layer, gt_agent_rasterizer, Rasterizer())
        # img = mtp_input_representation.make_input_representation(gt_instance_token, sample_token)

    prediction_results = []
    for token in prediction_challenge_tokens:
        pred = predictor(token)
        if pred != None:
            prediction_results.append(pred)


        if plot_gt:
            gt_instance_token, sample_token = token.split("_")
            gt_img = gt_representation.make_input_representation(gt_instance_token, sample_token)
            plt.imsave(prediction_plot_path + '/' + token + '_gt.png', gt_img)
            plt.close('all')

    prediction_dicts = [p.serialize() for p in prediction_results]
    fname = predction_results_path + '/' + split + '_tracking_results.json' 
    with open(fname, 'w') as f:
        json.dump(prediction_dicts, f)
        
        #break


if __name__ == '__main__':
    main()