from collections import defaultdict
from typing import List, Dict

from nuscenes.eval.common.data_classes import EvalBoxes
from sequential_perception.constants import NUSCENES_TRACKING_NAMES
import numpy as np
from filterpy.kalman import KalmanFilter, ExtendedKalmanFilter
from nuscenes import NuScenes
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.eval.tracking.data_classes import TrackingBox 
from nuscenes.eval.detection.data_classes import DetectionBox 
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
from scipy.optimize import linear_sum_assignment
from .iou3d import box3d_iou


iou_threshold = defaultdict(lambda : 0.01)
dist_threshold = defaultdict(lambda : 2.0)
#dist_threshold['bicycle'] = 2.0
# iou_threshold['pedestrian'] = 0.0
# iou_threshold['bicycle'] = 0.0

def greedy_matching(distance_matrix):
  '''
  Find the one-to-one matching using greedy allgorithm choosing small distance
  distance_matrix: (num_detections, num_tracks)
  '''
  matched_indices = []

  num_detections, num_tracks = distance_matrix.shape
  distance_1d = distance_matrix.reshape(-1)
  index_1d = np.argsort(distance_1d)
  index_2d = np.stack([index_1d // num_tracks, index_1d % num_tracks], axis=1)
  detection_id_matches_to_tracking_id = [-1] * num_detections
  tracking_id_matches_to_detection_id = [-1] * num_tracks
  for sort_i in range(index_2d.shape[0]):
    detection_id = int(index_2d[sort_i][0])
    tracking_id = int(index_2d[sort_i][1])
    if tracking_id_matches_to_detection_id[tracking_id] == -1 and detection_id_matches_to_tracking_id[detection_id] == -1:
      tracking_id_matches_to_detection_id[tracking_id] = detection_id
      detection_id_matches_to_tracking_id[detection_id] = tracking_id
      matched_indices.append([detection_id, tracking_id])

  matched_indices = np.array(matched_indices)
  return matched_indices


def hungarian_matching(cost_matrix):
    return np.array(linear_sum_assignment(cost_matrix)).T


def angle_in_pi(angle):
    if angle >= np.pi:
        angle -= 2*np.pi
    elif angle < -np.pi:
        angle += 2*np.pi
    return angle


class BoxKalmanFilter:
    def __init__(self, det0, track_id, tracking_name, score=0.1):
        self.dt = 0.5
        self.kf = KalmanFilter(dim_x=11, dim_z=7)
        self.kf.F = np.array([[1,0,0,0,0,0,0,1,0,0,0],      # state transition matrix
                            [0,1,0,0,0,0,0,0,1,0,0],
                            [0,0,1,0,0,0,0,0,0,1,0],
                            [0,0,0,1,0,0,0,0,0,0,1],  
                            [0,0,0,0,1,0,0,0,0,0,0],
                            [0,0,0,0,0,1,0,0,0,0,0],
                            [0,0,0,0,0,0,1,0,0,0,0],
                            [0,0,0,0,0,0,0,1,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0],
                            [0,0,0,0,0,0,0,0,0,1,0],
                            [0,0,0,0,0,0,0,0,0,0,1]])     
        
        self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0,0],      # measurement function,
                                [0,1,0,0,0,0,0,0,0,0,0],
                                [0,0,1,0,0,0,0,0,0,0,0],
                                [0,0,0,1,0,0,0,0,0,0,0],
                                [0,0,0,0,1,0,0,0,0,0,0],
                                [0,0,0,0,0,1,0,0,0,0,0],
                                [0,0,0,0,0,0,1,0,0,0,0]])
        
        # TODO Replace these with per-class values
        self.kf.P[:7:, :7] *= 10
        self.kf.P[7:,7:] *= 100
        #self.kf.Q[7:,7:] *= 0.1
        self.kf.R *= 0.1
        self.kf.Q *= 0.1

        pos0 = np.array(det0.translation)
        yaw0 = quaternion_yaw(Quaternion(det0.rotation))
        wlh0 = np.array(det0.size)
        vel0 = np.concatenate([det0.velocity, [0]]) * self.dt
        yawdot0 = 0
        x0 = np.concatenate([pos0, [yaw0], wlh0, vel0, [yawdot0]])
        self.kf.x[:, 0] = x0

        self.time_since_update = 0
        self.id = track_id
        self.history = []
        self.hits = 1           # number of total hits including the first detection
        self.hit_streak = 1     # number of continuing hit considering the first detection
        self.first_continuing_hit = 1
        self.still_first = True
        self.age = 0
        self.tracking_name = tracking_name
        self.score = score

    def predict(self):       
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        self.kf.predict()      
        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

        self.age += 1
        self.time_since_update += 1
        self.history.append(self.kf.x)
        return self.history[-1]

    def update(self, obs, score=0.1):
        self.score = score
        self.time_since_update = 0
        self.history = []
        self.hits += 1

        # Orientation correction
        self.kf.x[3] = angle_in_pi(self.kf.x[3])
        obs[3] = angle_in_pi(obs[3])

        if np.abs(self.kf.x[3] - obs[3]) > np.pi/2 and np.abs(self.kf.x[3] - obs[3]) < 3*np.pi/2:
            self.kf.x[3] += np.pi
            self.kf.x[3] = angle_in_pi(self.kf.x[3])
        
        if np.abs(self.kf.x[3] - obs[3]) >= 3*np.pi/2:
            if obs[3] > 0: self.kf.x[3] += np.pi *3/2
            else: self.kf.x[3] -= 2*np.pi        
        
        self.kf.update(obs)
        self.kf.x[3] = angle_in_pi(self.kf.x[3])
        self.history.append(self.kf.x)


        return self.history[-1]


class ConstVelTurnEKF:
    def __init__(self, det0, track_id, tracking_name, score=0.1) -> None:
        self.kf = ExtendedKalmanFilter(dim_x=9, dim_z=7)
        
        # TODO Replace these with per-class values
        # self.kf.P[:7:, :7] *= 10
        # self.kf.P[7:,7:] *= 100
        # self.kf.Q[7:,7:] *= 0.1
        # self.kf.R *= 1.0

        self.kf.P[:7:, :7] *= 10
        self.kf.P[7:,7:] *= 100
        #self.kf.Q[7:,7:] *= 0.1
        self.kf.R *= 0.1
        self.kf.Q *= 0.1

        pos0 = np.array(det0.translation)
        yaw0 = quaternion_yaw(Quaternion(det0.rotation))
        wlh0 = np.array(det0.size)
        vel0 = np.linalg.norm(det0.velocity)
        yawdot0 = 0
        x0 = np.concatenate([pos0, [yaw0], wlh0, [vel0], [yawdot0]])
        self.kf.x[:, 0] = x0

        self.time_since_update = 0
        self.id = track_id
        self.history = []
        self.hits = 1           # number of total hits including the first detection
        self.hit_streak = 1     # number of continuing hit considering the first detection
        self.first_continuing_hit = 1
        self.still_first = True
        self.age = 0
        self.tracking_name = tracking_name
        self.score = score
        self.dt = 0.5

    def dynamics(self, x):
        px = x[0]; py = x[1]; pz = x[2]
        theta = x[3]; omega = x[-1]
        w = x[4]; l = x[5]; h = x[6]
        v = x[7]
        dt = self.dt

        x_new = np.zeros(x.shape)
        x_new[0] = (v/omega)*np.sin(omega*dt+theta) - (v/omega)*np.sin(theta) + px
        x_new[1] = -(v/omega)*np.cos(omega*dt+theta) + (v/omega)*np.cos(theta) + py
        x_new[2] = pz
        x_new[3] = omega*dt+theta
        x_new[4] = w
        x_new[5] = l
        x_new[6] = h
        x_new[7] = v
        x_new[8] = omega

        return x_new

    def measurement(self, x):
        return np.squeeze(x)[:7].reshape(-1, 1)

    def dynamics_jacobian(self, x):
        #x = x.flatten()
        #print(x)
        px = x[0]; py = x[1]; pz = x[2]
        theta = x[3]; omega = x[-1]
        w = x[4]; l = x[5]; h = x[6]
        v = x[7]
        dt = self.dt

        F = np.eye(9)
        F[0, 3] = (v*np.cos(theta + dt*omega))/omega - (v*np.cos(theta))/omega
        F[0, 7] = np.sin(theta + dt*omega)/omega - np.sin(theta)/omega
        F[0, 8] = (v*np.sin(theta))/omega**2 - (v*np.sin(theta + dt*omega))/omega**2 + (dt*v*np.cos(theta + dt*omega))/omega
        F[1, 3] = (v*np.sin(theta + dt*omega))/omega - (v*np.sin(theta))/omega
        F[1, 7] = np.cos(theta)/omega - np.cos(theta + dt*omega)/omega
        F[1, 8] = (v*np.cos(theta + dt*omega))/omega**2 - (v*np.cos(theta))/omega**2 + (dt*v*np.sin(theta + dt*omega))/omega
        F[3, -1] = dt

        return F

    def measurement_jacobian(self, x):
        H = np.array([[1,0,0,0,0,0,0,0,0],
                      [0,1,0,0,0,0,0,0,0],
                      [0,0,1,0,0,0,0,0,0],
                      [0,0,0,1,0,0,0,0,0],
                      [0,0,0,0,1,0,0,0,0],
                      [0,0,0,0,0,1,0,0,0],
                      [0,0,0,0,0,0,1,0,0]])
        return H


    def predict(self):       
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if self.kf.x[-1] < 1e-3:
            self.kf.x[-1] += 1e-3

        # Update state transition matrix
        self.kf.F = self.dynamics_jacobian(self.kf.x)

        # Predict
        self.kf.predict()

        # Orientation correction      
        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

        self.age += 1
        self.time_since_update += 1
        self.history.append(self.kf.x)
        return self.history[-1]

    def update(self, obs, score=0.1):
        self.score = score
        self.time_since_update = 0
        self.history = []
        self.hits += 1

        # Orientation correction
        self.kf.x[3] = angle_in_pi(self.kf.x[3])
        obs[3] = angle_in_pi(obs[3])

        if np.abs(self.kf.x[3] - obs[3]) > np.pi/2 and np.abs(self.kf.x[3] - obs[3]) < 3*np.pi/2:
            self.kf.x[3] += np.pi
            self.kf.x[3] = angle_in_pi(self.kf.x[3])
        
        if np.abs(self.kf.x[3] - obs[3]) >= 3*np.pi/2:
            if obs[3] > 0: self.kf.x[3] += np.pi *3/2
            else: self.kf.x[3] -= 2*np.pi
        
        self.kf.update(obs.reshape(-1, 1), self.measurement_jacobian, self.measurement)
        self.kf.x[3] = angle_in_pi(self.kf.x[3])
        self.history.append(self.kf.x)


        return self.history[-1]
    

class Tracker:
    def __init__(self, tracking_name, gate_distance=2.0, dt=0.5) -> None:
        self.tracking_name = tracking_name
        self.tracks = []
        self.dist_threshold = gate_distance # dist_threshold[tracking_name]
        self.max_age = 2
        self.min_hits = 3
        self.dt = dt
        self.frame_count = 0
        self.track_count = 0
        id = 10000
        self.base_ids = {}
        for name in NUSCENES_TRACKING_NAMES:
            self.base_ids[name] = id
            id += 10000
    
    def create_track(self):
        pass

    def delete_track(self):
        pass
    
    def detection_to_box(self, det):
        box = Box(center=det.translation,
                  size=det.size,
                  orientation=Quaternion(det.rotation))
        return box

    def kfstate_to_box(self, x):
        x = np.squeeze(x)
        translation = x[:3]
        yaw = x[3]
        wlh = x[4:7]
        orientation = Quaternion(axis=[0, 0, 1], angle=yaw)
        box = Box(center=translation,
                  size=wlh,
                  orientation=orientation)
        return box

    def box_to_kfobs(self, box):
        translation = box.center
        theta = quaternion_yaw(box.orientation)
        size = box.wlh
        obs = np.concatenate([translation, [theta], size])
        return obs


    def get_tracking_result(self, track, token):
        x = np.squeeze(track.kf.x)
        translation = x[:3]
        yaw = x[3]
        rotation = Quaternion(axis=[0, 0, 1], angle=yaw)
        size = x[4:7]
        #vel = x[7:10]
        sample_result = {
            'sample_token': token,
            'translation': translation.tolist(),
            'size': size.tolist(),
            'rotation': rotation.elements.tolist(),
            #'velocity': (vel[:2]/self.dt).tolist(),
            'velocity': [0, 0],
            'tracking_id': str(int(track.id)),
            'tracking_name': self.tracking_name,
            'tracking_score': track.score #0.1
        }
        return sample_result

    def get_iou(self, detections, tracks):
        ndets = len(detections)
        ntracks = len(tracks)
        iou_matrix = np.zeros((ndets, ntracks))

        for i in range(ndets):
            det = detections[i]
            det_corners = det.corners().T
            for j in range(ntracks):
                track = tracks[j]
                track_corners = track.corners().T
                iou_matrix[i, j] = box3d_iou(det_corners, track_corners)[0]

        return iou_matrix

    def get_cartesian_distance(self, detections, tracks):
        ndets = len(detections)
        ntracks = len(tracks)
        dist_matrix = np.zeros((ndets, ntracks))

        for i in range(ndets):
            det = detections[i]
            for j in range(ntracks):
                track = tracks[j]
                dist_matrix[i, j] = np.linalg.norm(det.center - track.center)

        return dist_matrix

    def associate_by_distance(self, detections, tracks):
        """
        Associate detections by cartesian distance
        """
        if len(tracks) == 0 or len(detections) == 0:
            matched_idxs = np.empty((0,2),dtype=int)
            unmatched_dets = np.arange(len(detections))
            unmatched_tracks = np.empty((0,1),dtype=int)  
            return matched_idxs, unmatched_dets, unmatched_tracks

        metric_matrix = self.get_cartesian_distance(detections, tracks)
        distance_matrix = metric_matrix

        #matched_indices = hungarian_matching(distance_matrix)
        matched_indices = greedy_matching(distance_matrix)

        unmatched_detections = []
        for d,det in enumerate(detections):
            if(d not in matched_indices[:,0]):
                unmatched_detections.append(d)

        unmatched_trackers = []
        for t,trk in enumerate(tracks):
            if len(matched_indices) == 0 or (t not in matched_indices[:,1]):
                unmatched_trackers.append(t)

        matches = []
        for m in matched_indices:
            match = True
            if(metric_matrix[m[0],m[1]]>self.dist_threshold):
                match = False
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1,2))

        matches = np.concatenate(matches, axis=0) if len(matches)>0 else np.empty((0,2), dtype=int)
        

        return matches, unmatched_detections, unmatched_trackers

    def update(self, detections, sample_token=None):
        if sample_token == None:
            sample_token == detections[0].sample_token

        self.frame_count += 1
        ## Predict ##
        predicted_states = [kf.predict() for kf in self.tracks]
        predicted_boxes = [self.kfstate_to_box(x) for x in predicted_states]

        ## Associate ##
        detected_boxes = [self.detection_to_box(det) for det in detections]
        # matched_idxs, unmatched_dets, unmatched_tracks = self.associate(detected_boxes, predicted_boxes)
        matched_idxs, unmatched_dets, unmatched_tracks = self.associate_by_distance(detected_boxes, predicted_boxes)

        ## Update ##
        # update matched tracks with assigned detections
        for det_idx, track_idx in matched_idxs:
            obs = self.box_to_kfobs(detected_boxes[det_idx])
            score = detections[det_idx].detection_score
            self.tracks[track_idx].update(obs, score)

        # create new tracks for unmatched detections
        for idx in unmatched_dets:
            #self.create_track(detections[i])
            
            det0 = detections[idx]
            self.track_count += 1
            new_id = self.base_ids[self.tracking_name] + self.track_count
            
            score = detections[idx].detection_score
            track = BoxKalmanFilter(det0, new_id, self.tracking_name, score=score)
            #track = ConstVelTurnEKF(det0, new_id, self.tracking_name, score=score)
            self.tracks.append(track)

        # Determine which tracks should be returned or deleted
        tracks_to_delete = []
        tracks_to_return = []
        tracking_results = []
        for i in range(len(self.tracks)):
            track = self.tracks[i]
            if track.time_since_update < self.max_age:
                if track.hits > self.min_hits or self.frame_count <= self.min_hits:
                    result = self.get_tracking_result(self.tracks[i], sample_token)
                    tracking_results.append(result)
            else:
                tracks_to_delete.append(i)

        # Delete the stale tracks
        for idx in sorted(tracks_to_delete, reverse=True):
            self.tracks.pop(idx)

        return tracking_results
        
        
        # for idx in tracks_to_return:
        #     result = self.get_tracking_result(self.tracks[i], detections[0].sample_token)
        #     tracking_results.append(result)
            


class MultiObjectTrackByDetection:
    def __init__(self,
                 nuscenes: NuScenes,
                 tracking_names: List = NUSCENES_TRACKING_NAMES,
                 gate_dists: Dict = defaultdict(lambda : 2.0)):

        self.nuscenes = nuscenes
        self.tracking_names = tracking_names
        self.gate_dists= gate_dists
        self.trackers = {tracking_name: Tracker(tracking_name, self.gate_dists[tracking_name]) for tracking_name in self.tracking_names}
        return

    def reset(self):
        self.tracking_results = {}
        self.trackers = {tracking_name: Tracker(tracking_name, self.gate_dists[tracking_name]) for tracking_name in self.tracking_names}

    def __call__(self, detection_data: Dict, reset=True):

        all_results = EvalBoxes.deserialize(detection_data['results'], DetectionBox)
        det_meta = detection_data['meta']

        #print("meta: ", det_meta)
        #print("Found detections for {} samples.".format(len(all_results.sample_tokens)))

        # Collect tokens for all scenes in results
        scene_tokens = []
        for sample_token in all_results.sample_tokens:
            scene_token = self.nuscenes.get('sample', sample_token)['scene_token']
            if scene_token not in scene_tokens:
                scene_tokens.append(scene_token)

        tracking_results = {}

        for scene_token in scene_tokens:
            current_sample_token = self.nuscenes.get('scene', scene_token)['first_sample_token']

            #trackers = {tracking_name: Tracker(tracking_name) for tracking_name in self.tracking_name}

            while current_sample_token != '' and current_sample_token in all_results.sample_tokens:
                tracking_results[current_sample_token] = []

                # Form detection observation:  [x, y, z, angle, l, h, w]
                # Group observations by detection name
                dets = {tracking_name: [] for tracking_name in self.tracking_names}
                info = {tracking_name: [] for tracking_name in self.tracking_names}
                for box in all_results.boxes[current_sample_token]:
                    if box.detection_name not in self.tracking_names:
                        continue
                    detection = box
                    information = np.array([box.detection_score])
                    dets[box.detection_name].append(detection)
                    info[box.detection_name].append(information)


                # Update Tracker
                for tracking_name in self.tracking_names:
                    updated_tracks = self.trackers[tracking_name].update(dets[tracking_name], sample_token=current_sample_token)
                    tracking_results[current_sample_token] += updated_tracks

                current_sample_token = self.nuscenes.get('sample', current_sample_token)['next']
                
        tracking_meta = {
            "use_camera":   False,
            "use_lidar":    True,
            "use_radar":    False,
            "use_map":      False,
            "use_external": False,
        }

        tracking_output_data = {'meta': tracking_meta, 'results': tracking_results}

        if reset:
            self.reset()

        return tracking_output_data

    def update(self, detection_results: Dict):
        "Update tracker with results from one sample"

        # all_results = EvalBoxes.deserialize(detection_data['results'], DetectionBox)
        all_results = EvalBoxes.deserialize(detection_results, DetectionBox)
        assert len(all_results.sample_tokens)==1, 'Function update() can only be called on results from one sample.'
        # det_meta = detection_data['meta']
        # scene_token = all_results.sample_tokens[0]

        #print("meta: ", det_meta)
        #print("Found detections for {} samples.".format(len(all_results.sample_tokens)))

        # Collect tokens for all scenes in results
        # scene_tokens = []
        # for sample_token in all_results.sample_tokens:
        #     scene_token = self.nuscenes.get('sample', sample_token)['scene_token']
        #     if scene_token not in scene_tokens:
        #         scene_tokens.append(scene_token)

        #tracking_results = {}
        # all_sample_tokens = all_results.sample_tokens()
        # for scene_token in scene_tokens:

            # find first sample token in input detections
            # current_sample_token = self.nuscenes.get('scene', scene_token)['first_sample_token']
            # while current_sample_token not in all_sample_tokens:
            #     current_sample_token = self.nuscenes.get('sample', current_sample_token)['next']

            # #trackers = {tracking_name: Tracker(tracking_name) for tracking_name in self.tracking_name}

            # while current_sample_token != '':
        current_sample_token = all_results.sample_tokens[0]
        self.tracking_results[current_sample_token] = []

        # Form detection observation:  [x, y, z, angle, l, h, w]
        # Group observations by detection name
        dets = {tracking_name: [] for tracking_name in self.tracking_names}
        info = {tracking_name: [] for tracking_name in self.tracking_names}
        for box in all_results.boxes[current_sample_token]:
            if box.detection_name not in self.tracking_names:
                continue
            detection = box
            information = np.array([box.detection_score])
            dets[box.detection_name].append(detection)
            info[box.detection_name].append(information)


        # Update Trackers
        for tracking_name in self.tracking_names:
            updated_tracks = self.trackers[tracking_name].update(dets[tracking_name], sample_token=current_sample_token)
            self.tracking_results[current_sample_token] += updated_tracks


        # current_sample_token = self.nuscenes.get('sample', current_sample_token)['next']
                


        tracking_meta = {
            "use_camera":   False,
            "use_lidar":    True,
            "use_radar":    False,
            "use_map":      False,
            "use_external": False,
        }

        tracking_output_data = {'meta': tracking_meta, 'results': self.tracking_results}

        return tracking_output_data


    # def update(self, detection_data: Dict):

    #     all_results = EvalBoxes.deserialize(detection_data['results'], DetectionBox)
    #     det_meta = detection_data['meta']

    #     #print("meta: ", det_meta)
    #     #print("Found detections for {} samples.".format(len(all_results.sample_tokens)))

    #     # Collect tokens for all scenes in results
    #     scene_tokens = []
    #     for sample_token in all_results.sample_tokens:
    #         scene_token = self.nuscenes.get('sample', sample_token)['scene_token']
    #         if scene_token not in scene_tokens:
    #             scene_tokens.append(scene_token)

    #     #tracking_results = {}
    #     all_sample_tokens = all_results.sample_tokens()
    #     for scene_token in scene_tokens:

    #         # find first sample token in input detections
    #         current_sample_token = self.nuscenes.get('scene', scene_token)['first_sample_token']
    #         while current_sample_token not in all_sample_tokens:
    #             current_sample_token = self.nuscenes.get('sample', current_sample_token)['next']

    #         #trackers = {tracking_name: Tracker(tracking_name) for tracking_name in self.tracking_name}

    #         while current_sample_token != '':
    #             self.tracking_results[current_sample_token] = []

    #             # Form detection observation:  [x, y, z, angle, l, h, w]
    #             # Group observations by detection name
    #             dets = {tracking_name: [] for tracking_name in self.tracking_names}
    #             info = {tracking_name: [] for tracking_name in self.tracking_names}
    #             for box in all_results.boxes[current_sample_token]:
    #                 if box.detection_name not in self.tracking_names:
    #                     continue
    #                 detection = box
    #                 information = np.array([box.detection_score])
    #                 dets[box.detection_name].append(detection)
    #                 info[box.detection_name].append(information)


    #             # Update Tracker
    #             for tracking_name in self.tracking_names:
    #                 updated_tracks = self.trackers[tracking_name].update(dets[tracking_name])
    #                 self.tracking_results[current_sample_token] += updated_tracks


    #             current_sample_token = self.nuscenes.get('sample', current_sample_token)['next']
                


    #     tracking_meta = {
    #         "use_camera":   False,
    #         "use_lidar":    True,
    #         "use_radar":    False,
    #         "use_map":      False,
    #         "use_external": False,
    #     }

    #     tracking_output_data = {'meta': tracking_meta, 'results': self.tracking_results}

    #     return tracking_output_data
