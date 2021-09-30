from typing import Any, Dict, List, Tuple, Callable, Union
import numpy as np
from nuscenes import NuScenes
from nuscenes.prediction.helper import BUFFER, Record, convert_global_coords_to_local
from nuscenes.eval.common.utils import quaternion_yaw, angle_diff
from pyquaternion import Quaternion



class TrackingResultsPredictHelper:
    def __init__(self, nusc: NuScenes, tracking_results:Dict):
        self.data = nusc
        self.tracking_results = tracking_results

    def _timestamp_for_sample(self, sample_token: str) -> float:
        """
        Gets timestamp from sample token.
        :param sample_token: Get the timestamp for this sample.
        :return: Timestamp (microseconds).
        """
        return self.data.get('sample', sample_token)['timestamp']

    def _absolute_time_diff(self, time1: float, time2: float) -> float:
        """
        Helper to compute how much time has elapsed in _iterate method.
        :param time1: First timestamp (microseconds since unix epoch).
        :param time2: Second timestamp (microseconds since unix epoch).
        :return: Absolute Time difference in floats.
        """
        return abs(time1 - time2) / 1e6

    def _iterate(self, starting_annotation: Dict[str, Any], seconds: float, direction: str) -> List[Dict[str, Any]]:
        """
        Iterates forwards or backwards in time through the annotations for a given amount of seconds.
        :param starting_annotation: Sample annotation record to start from.
        :param seconds: Number of seconds to iterate.
        :param direction: 'prev' for past and 'next' for future.
        :return: List of annotations ordered by time.
        """
        if seconds < 0:
            raise ValueError(f"Parameter seconds must be non-negative. Received {seconds}.")

        # Need to exit early because we technically _could_ return data in this case if
        # the first observation is within the BUFFER.
        if seconds == 0:
            return []

        seconds_with_buffer = seconds + BUFFER
        starting_time = self._timestamp_for_sample(starting_annotation['sample_token'])

        next_annotation = starting_annotation
        gt_sample = self.data.get('sample', next_annotation['sample_token'])

        time_elapsed = 0.

        annotations = []

        expected_samples_per_sec = 2
        max_annotations = int(expected_samples_per_sec * seconds)
        while time_elapsed <= seconds_with_buffer and len(annotations) < max_annotations:

            # if next_annotation[direction] == '':
            #     break
            if gt_sample[direction] == '':
                break
            
            #gt_sample = self.data.get('sample', next_annotation['sample_token'])
            next_sample_token = gt_sample[direction]

            #next_annotation = self.data.get('sample_annotation', next_annotation[direction])
            #current_time = self._timestamp_for_sample(next_annotation['sample_token'])

            next_annotation_list = self.tracking_results[next_sample_token]
            end_track = True
            for tracking_result in next_annotation_list:
                #if starting_annotation['tracking_id'] in tracking_result.keys():
                if starting_annotation['tracking_id'] == tracking_result['tracking_id']:
                    next_annotation = tracking_result
                    end_track = False
                    break

            if end_track:
                break
            
            current_time = self._timestamp_for_sample(next_annotation['sample_token'])
            time_elapsed = self._absolute_time_diff(current_time, starting_time)
            gt_sample = self.data.get('sample', next_annotation['sample_token'])

            if time_elapsed < seconds_with_buffer:
                annotations.append(next_annotation)

        return annotations

    def _get_past_or_future_for_agent(self, instance_token: str, sample_token: str,
                                      seconds: float, in_agent_frame: bool,
                                      direction: str,
                                      just_xy: bool = True) -> Union[List[Record], np.ndarray]:
        """
        Helper function to reduce code duplication between get_future and get_past for agent.
        :param instance_token: Instance of token.
        :param sample_token: Sample token for instance.
        :param seconds: How many seconds of data to retrieve.
        :param in_agent_frame: Whether to rotate the coordinates so the
            heading is aligned with the y-axis.
        :param direction: 'next' for future or 'prev' for past.
        :return: array of shape [n_timesteps, 2].
        """
        #starting_annotation = self.get_sample_annotation(instance_token, sample_token)
        annotation_list =  self.tracking_results[sample_token]
        for ann in annotation_list:
            if ann['tracking_id'] == instance_token:
                starting_annotation = ann
                break
        
        sequence = self._iterate(starting_annotation, seconds, direction)

        if not just_xy:
            return sequence

        coords = np.array([r['translation'][:2] for r in sequence])

        if coords.size == 0:
            return coords

        if in_agent_frame:
            coords = convert_global_coords_to_local(coords,
                                                    starting_annotation['translation'],
                                                    starting_annotation['rotation'])

        return coords

    def get_past_for_agent(self, instance_token: str, sample_token: str,
                           seconds: float, in_agent_frame: bool,
                           just_xy: bool = True) -> Union[List[Record], np.ndarray]:
        """
        Retrieves the agent's past sample annotation records.
        :param instance_token: Instance token.
        :param sample_token: Sample token.
        :param seconds: How much past data to retrieve.
        :param in_agent_frame: If true, locations are rotated to the agent frame.
            Only relevant if just_xy = True.
        :param just_xy: If true, returns an np.array of x,y locations as opposed to the
            entire record.
        :return: If just_xy, np.ndarray. Else, List of records.
            The rows decrease with time, i.e the last row occurs the farthest in the past.
        """
        return self._get_past_or_future_for_agent(instance_token, sample_token, seconds,
                                                  in_agent_frame, direction='prev', just_xy=just_xy)


    def _get_past_or_future_for_sample(self, sample_token: str, seconds: float, in_agent_frame: bool,
                                       direction: str, just_xy: bool,
                                       function: Callable[[str, str, float, bool, str, bool], np.ndarray]) -> Union[Dict[str, np.ndarray], Dict[str, List[Record]]]:
        """
        Helper function to reduce code duplication between get_future and get_past for sample.
        :param sample_token: Sample token.
        :param seconds: How much past or future data to retrieve.
        :param in_agent_frame: Whether to rotate each agent future.
            Only relevant if just_xy = True.
        :param just_xy: If true, returns an np.array of x,y locations as opposed to the
            entire record.
        :param function: _get_past_or_future_for_agent.
        :return: Dictionary mapping instance token to np.array or list of records.
        """
        #sample_record = self.data.get('sample', sample_token)
        sample_records = self.tracking_results[sample_token]
        sequences = {}
        #for annotation in sample_record['anns']:
        for record in sample_records:
            #annotation_record = self.data.get('sample_annotation', annotation)
            sequence = function(record['tracking_id'],
                                record['sample_token'],
                                seconds, in_agent_frame, direction, just_xy=just_xy)

            sequences[record['tracking_id']] = sequence

        return sequences

    def get_past_for_sample(self, sample_token: str, seconds: float, in_agent_frame: bool,
                            just_xy: bool = True) -> Dict[str, np.ndarray]:
        """
        Retrieves the the past x,y locations of all agents in the sample.
        :param sample_token: Sample token.
        :param seconds: How much past data to retrieve.
        :param in_agent_frame: If true, locations are rotated to the agent frame.
                Only relevant if just_xy = True.
        :param just_xy: If true, returns an np.array of x,y locations as opposed to the
            entire record.
        :return: If just_xy, Mapping of instance token to np.ndarray.
            Else, the mapping is from instance token to list of records.
            The rows decrease with time, i.e the last row occurs the farthest in the past.
        """
        return self._get_past_or_future_for_sample(sample_token, seconds, in_agent_frame, 'prev',
                                                   just_xy,
                                                   function=self._get_past_or_future_for_agent)

    def get_annotations_for_sample(self, sample_token: str) -> List[Record]:
        """
        Gets a list of sample annotation records for a sample.
        :param sample_token: Sample token.
        """

        # sample_record = self.data.get('sample', sample_token)
        # annotations = []

        # for annotation_token in sample_record['anns']:
        #     annotation_record = self.data.get('sample_annotation', annotation_token)
        #     annotations.append(annotation_record)

        return self.tracking_results[sample_token]

    def get_sample_annotation(self, instance_token: str, sample_token: str) -> Record:
        """
        Retrieves an annotation given an instance token and its sample.
        :param instance_token: Instance token.
        :param sample_token: Sample token for instance.
        :return: Sample annotation record.
        """
        #return self.data.get('sample_annotation', self.inst_sample_to_ann[(sample_token, instance_token)])
        for ann in self.tracking_results[sample_token]:
            if ann['tracking_id'] == instance_token:
                return ann

    def _compute_diff_between_sample_annotations(self, instance_token: str,
                                                 sample_token: str, max_time_diff: float,
                                                 with_function, **kwargs) -> float:
        """
        Grabs current and previous annotation and computes a float from them.
        :param instance_token: Instance token.
        :param sample_token: Sample token.
        :param max_time_diff: If the time difference between now and the most recent annotation is larger
            than this param, function will return np.nan.
        :param with_function: Function to apply to the annotations.
        :param **kwargs: Keyword arguments to give to with_function.
        """
        annotation = self.get_sample_annotation(instance_token, sample_token)
        history = self.get_past_for_agent(instance_token, sample_token, max_time_diff, in_agent_frame=False, just_xy=False)

        if len(history)<1:
            return np.nan

        prev = history[0]

        # if annotation['prev'] == '':
        #     return np.nan

        #prev = self.data.get('sample_annotation', annotation['prev'])

        current_time = 1e-6 * self.data.get('sample', sample_token)['timestamp']
        prev_time = 1e-6 * self.data.get('sample', prev['sample_token'])['timestamp']
        time_diff = current_time - prev_time

        if time_diff <= max_time_diff:

            return with_function(annotation, prev, time_diff, **kwargs)

        else:
            return np.nan

    def get_velocity_for_agent(self, instance_token: str, sample_token: str, max_time_diff: float = 1.5) -> float:
        """
        Computes velocity based on the difference between the current and previous annotation.
        :param instance_token: Instance token.
        :param sample_token: Sample token.
        :param max_time_diff: If the time difference between now and the most recent annotation is larger
            than this param, function will return np.nan.
        """
        return self._compute_diff_between_sample_annotations(instance_token, sample_token, max_time_diff,
                                                             with_function=self.velocity)

    def get_heading_change_rate_for_agent(self, instance_token: str, sample_token: str,
                                          max_time_diff: float = 1.5) -> float:
        """
        Computes heading change rate based on the difference between the current and previous annotation.
        :param instance_token: Instance token.
        :param sample_token: Sample token.
        :param max_time_diff: If the time difference between now and the most recent annotation is larger
            than this param, function will return np.nan.
        """
        return self._compute_diff_between_sample_annotations(instance_token, sample_token, max_time_diff,
                                                             with_function=self.heading_change_rate)

    def get_acceleration_for_agent(self, instance_token: str, sample_token: str, max_time_diff: float = 1.5) -> float:
        """
        Computes heading change rate based on the difference between the current and previous annotation.
        :param instance_token: Instance token.
        :param sample_token: Sample token.
        :param max_time_diff: If the time difference between now and the most recent annotation is larger
            than this param, function will return np.nan.
        """
        return self._compute_diff_between_sample_annotations(instance_token, sample_token,
                                                             max_time_diff,
                                                             with_function=self.acceleration,
                                                             instance_token_for_velocity=instance_token)

    def velocity(self, current: Dict[str, Any], prev: Dict[str, Any], time_diff: float) -> float:
        """
        Helper function to compute velocity between sample annotations.
        :param current: Sample annotation record for the current timestamp.
        :param prev: Sample annotation record for the previous time stamp.
        :param time_diff: How much time has elapsed between the records.
        """
        diff = (np.array(current['translation']) - np.array(prev['translation'])) / time_diff
        return np.linalg.norm(diff[:2])

    def heading_change_rate(self, current: Dict[str, Any], prev: Dict[str, Any], time_diff: float) -> float:
        """
        Helper function to compute heading change rate between sample annotations.
        :param current: Sample annotation record for the current timestamp.
        :param prev: Sample annotation record for the previous time stamp.
        :param time_diff: How much time has elapsed between the records.
        """
        current_yaw = quaternion_yaw(Quaternion(current['rotation']))
        prev_yaw = quaternion_yaw(Quaternion(prev['rotation']))

        return angle_diff(current_yaw, prev_yaw, period=2*np.pi) / time_diff

    def acceleration(self, current: Dict[str, Any], prev: Dict[str, Any],
                    time_diff: float, instance_token_for_velocity: str) -> float:
        """
        Helper function to compute acceleration between sample annotations.
        :param current: Sample annotation record for the current timestamp.
        :param prev: Sample annotation record for the previous time stamp.
        :param time_diff: How much time has elapsed between the records.
        :param instance_token_for_velocity: Instance token to compute velocity.
        :param helper: Instance of PredictHelper.
        """
        current_velocity = self.get_velocity_for_agent(instance_token_for_velocity, current['sample_token'])
        prev_velocity = self.get_velocity_for_agent(instance_token_for_velocity, prev['sample_token'])

        return (current_velocity - prev_velocity) / time_diff