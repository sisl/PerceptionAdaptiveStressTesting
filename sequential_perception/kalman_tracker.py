from collections import defaultdict
from sequential_perception.constants import NUSCENES_TRACKING_NAMES
import numpy as np
from filterpy.kalman import KalmanFilter, ExtendedKalmanFilter
from numpy.core.defchararray import translate
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.eval.tracking.data_classes import TrackingBox 
from nuscenes.eval.detection.data_classes import DetectionBox 
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
from scipy.optimize import linear_sum_assignment
#from scipy.spatial.kdtree import distance_matrix
from .iou3d import evalbox_iou3d, box3d_iou


iou_threshold = defaultdict(lambda : 0.01)
dist_threshold = defaultdict(lambda : 10.0)
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
    def __init__(self, x0, track_id, tracking_name, score=0.1) -> None:
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
        self.kf.Q[7:,7:] *= 0.1
        self.kf.R *= 1.0

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
    


class Tracker:
    def __init__(self, tracking_name) -> None:
        self.tracking_name = tracking_name
        self.tracks = []
        self.dist_threshold = dist_threshold[tracking_name]
        self.max_age = 2
        self.min_hits = 3
        self.dt = 0.5
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
        vel = x[7:10]
        sample_result = {
            'sample_token': token,
            'translation': translation.tolist(),
            'size': size.tolist(),
            'rotation': rotation.elements.tolist(),
            'velocity': (vel[:2]/self.dt).tolist(),
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

        #distance_matrix[distance_matrix > self.dist_threshold] += np.max(distance_matrix)

        matched_indices = hungarian_matching(distance_matrix)
        #matched_indices = greedy_matching(distance_matrix)

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

    def update(self, detections):
        if self.tracking_name == 'bicycle':
            print('DEBUG')


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
            pos0 = detected_boxes[idx].center
            yaw0 = quaternion_yaw(detected_boxes[idx].orientation)
            wlh0 = detected_boxes[idx].wlh
            vel0 = np.concatenate([detections[idx].velocity, [0]]) * self.dt
            yawdot0 = 0
            x0 = np.concatenate([pos0, [yaw0], wlh0, vel0, [yawdot0]])
            
            self.track_count += 1
            new_id = self.base_ids[self.tracking_name] + self.track_count
            
            score = detections[idx].detection_score
            track = BoxKalmanFilter(x0, new_id, self.tracking_name, score=score)
            self.tracks.append(track)

        # Determine which tracks should be returned or deleted
        tracks_to_delete = []
        tracks_to_return = []
        tracking_results = []
        for i in range(len(self.tracks)):
            track = self.tracks[i]
            if track.time_since_update < self.max_age:
                if track.hits > self.min_hits or self.frame_count <= self.min_hits:
                    result = self.get_tracking_result(self.tracks[i], detections[0].sample_token)
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
            



