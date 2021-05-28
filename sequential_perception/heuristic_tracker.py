import numpy as np
from collections import defaultdict
from .constants import NUSCENES_TRACKING_NAMES

"""
class Tracker

Keeps a list of all detected objects and, ego vehicle gps information.
Then for each new object added to the tracker, checks if that object was seen
in the previous timestep. If the new object is the same class and could feasibly have 
moved to the new spot in one timestep they are assumed to be the same object.
"""

class HeuristicTracker():
    def __init__(self, tracking_name):
        #self.options = 12
        #self.colors = [np.array(colorsys.hsv_to_rgb(i/self.options, 1, 1))*255 for i in range(self.options)]
        self.colorIndex = 0
        #self.currentTime = -1
        self.currentTime = 0
        self.histories = []
        self.pastCars = []
        self.pastPeds = []
        self.pastBikes = []
        self.trackedGPS = None
        self.scale = None
        self.offsets = []
        self.tracking_name = tracking_name

        self.tracks = []
        self.track_ids = []

        self.duplicated_thresh = defaultdict(lambda : 1.0)
        self.duplicated_thresh['pedestrian'] = 0.42
        self.duplicated_thresh['bicycle'] = 0.56

        self.dist_thresh = defaultdict(lambda : 0.5)
        self.dist_thresh['pedestrian'] = 0.5
        self.dist_thresh['bicycle'] = 0.5

        id = 10000
        self.base_ids = {}
        for name in NUSCENES_TRACKING_NAMES:
            self.base_ids[name] = id
            id += 10000

        self.track_count = 0

        #self.duplicated_thresh = {'pedestrian':0.42, 'bicycle':0.56}

    def get_det_distance(self, det1, det2):
        r1 = np.array(det1.translation)
        r2 = np.array(det2.translation)
        return np.linalg.norm(r1 - r2)

    def create_new_track(self, det):
        self.track_count += 1
        self.tracks.append([det])
        new_id = self.base_ids[self.tracking_name] + self.track_count
        self.track_ids.append(new_id)
        return new_id


    # Update tracks 
    def update(self, detections):
        
        # Remove possible duplicates
        n_dets = len(detections)
        is_duplicated =  n_dets*[False]
        dets = []
        for i in range(n_dets):
            for j in range(i+1, n_dets):
                #dist = np.linalg.norm(detections[i].translation - detections[j].translation)
                dist = self.get_det_distance(detections[i], detections[j])
                if dist < self.duplicated_thresh[self.tracking_name]:
                    is_duplicated[i] = True
            if not is_duplicated[i]:
                dets.append(detections[i])
        
        # Match each detection to existing objects
        # Create new tracks
        # Delete old tracks
        updated_tracks = []

        for det in dets:

            # get distance from new detection to most recent detection in each track
            dists = [self.get_det_distance(det, t[-1]) for t in self.tracks]
            if len(dists) == 0:
                id = self.create_new_track(det)
                updated_tracks.append({'track':det, 'track_id':id})
                continue

            
            closest_index = np.argsort(dists)[0]
            if dists[closest_index] <= self.dist_thresh[self.tracking_name]:
                self.tracks[closest_index].append(det)
                updated_tracks.append({'track':det, 'track_id':self.track_ids[closest_index]})

            else:
                id = self.create_new_track(det)
                updated_tracks.append({'track':det, 'track_id':id})

        return updated_tracks