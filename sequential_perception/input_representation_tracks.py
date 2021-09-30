from typing import Any, Dict, List, Tuple, Callable, Union
import cv2
import numpy as np
from nuscenes.prediction.helper import angle_of_rotation
from pyquaternion import Quaternion
from nuscenes.prediction.helper import quaternion_yaw
from nuscenes.prediction import PredictHelper
from nuscenes.prediction.input_representation.interface import AgentRepresentation

from nuscenes.prediction.input_representation.agents import History, get_track_box, reverse_history,  fade_color, default_colors
from nuscenes.prediction.input_representation.utils import get_crops, get_rotation_matrix

from nuscenes.prediction.input_representation.static_layers import Color, change_color_of_binary_mask, correct_yaw, draw_lanes_in_agent_frame, get_patchbox
from nuscenes.prediction.input_representation.combinators import Rasterizer

from nuscenes.prediction.input_representation.interface import StaticLayerRepresentation
from nuscenes.prediction.input_representation.static_layers import load_all_maps


from .predict_helper_tracks import TrackingResultsPredictHelper

class AgentBoxesFromTracking(AgentRepresentation):
    """
    Represents the past sequence of agent states as a three-channel
    image with faded 2d boxes.
    """

    def __init__(self, tracking_helper: TrackingResultsPredictHelper,
                 seconds_of_history: float = 2,
                 frequency_in_hz: float = 2,
                 resolution: float = 0.1,  # meters / pixel
                 meters_ahead: float = 40, meters_behind: float = 10,
                 meters_left: float = 25, meters_right: float = 25,
                 color_mapping: Callable[[str], Tuple[int, int, int]] = None):

        self.track_helper = tracking_helper
        self.nusc_helper = PredictHelper(tracking_helper.data)
        self.seconds_of_history = seconds_of_history
        self.frequency_in_hz = frequency_in_hz
        #self.tracking_results = tracking_results

        if not resolution > 0:
            raise ValueError(f"Resolution must be positive. Received {resolution}.")

        self.resolution = resolution

        self.meters_ahead = meters_ahead
        self.meters_behind = meters_behind
        self.meters_left = meters_left
        self.meters_right = meters_right

        if not color_mapping:
            color_mapping = default_colors

        self.color_mapping = color_mapping

    def add_present_time_to_history(self, current_time: List[Dict[str, Any]],
                                    history: History) -> History:
        """
        Adds the sample annotation records from the current time to the
        history object.
        :param current_time: List of sample annotation records from the
            current time. Result of get_annotations_for_sample method of
            PredictHelper.
        :param history: Result of get_past_for_sample method of PredictHelper.
        :return: History with values from current_time appended.
        """

        for annotation in current_time:
            token = annotation['tracking_id']

            if token in history:

                # We append because we've reversed the history
                history[token].append(annotation)

            else:
                history[token] = [annotation]

        return history

    def color_map(self, tracking_name: str) -> Tuple[int, int, int]:
        """
        Maps a category name to an rgb color (without fading).
        :param category_name: Name of object category for the annotation.
        :return: Tuple representing rgb color.
        """

        if tracking_name in ['car', 'bus', 'truck', 'bicycle', 'motorcycle', 'trailer']:
            return 255, 255, 0  # yellow
        # elif tracking_name in ['trailer']:
        #     return 204, 0, 204  # violet
        elif tracking_name in ['pedestrian']:
            return 255, 153, 51  # orange
        else:
            raise ValueError(f"Cannot map {tracking_name} to a color.")

    def draw_agent_boxes(self, center_agent_annotation: Dict[str, Any],
                        center_agent_pixels: Tuple[float, float],
                        agent_history: History,
                        base_image: np.ndarray,
                        get_color: Callable[[str], Tuple[int, int, int]],
                        resolution: float = 0.1) -> None:
        """
        Draws past sequence of agent boxes on the image.
        :param center_agent_annotation: Annotation record for the agent
            that is in the center of the image.
        :param center_agent_pixels: Pixel location of the agent in the
            center of the image.
        :param agent_history: History for all agents in the scene.
        :param base_image: Image to draw the agents in.
        :param get_color: Mapping from category_name to RGB tuple.
        :param resolution: Size of the image in pixels / meter.
        :return: None.
        """

        agent_x, agent_y = center_agent_annotation['translation'][:2]

        for instance_token, annotations in agent_history.items():

            num_points = len(annotations)

            for i, annotation in enumerate(annotations):

                box = get_track_box(annotation, (agent_x, agent_y), center_agent_pixels, resolution)

                if instance_token == center_agent_annotation['tracking_id']:
                    color = (255, 0, 0)
                else:
                    color = get_color(annotation['tracking_name'])

                # Don't fade the colors if there is no history
                if num_points > 1:
                    color = fade_color(color, i, num_points - 1)

                cv2.fillPoly(base_image, pts=[np.int0(box)], color=color)

    def make_representation(self, instance_token: str, sample_token: str) -> np.ndarray:
        """
        Draws agent boxes with faded history into a black background.
        :param instance_token: Instance token.
        :param sample_token: Sample token.
        :return: np.ndarray representing a 3 channel image.
        """

        # Taking radius around track before to ensure all actors are in image
        buffer = max([self.meters_ahead, self.meters_behind,
                      self.meters_left, self.meters_right]) * 2

        image_side_length = int(buffer/self.resolution)

        # We will center the track in the image
        central_track_pixels = (image_side_length / 2, image_side_length / 2)

        base_image = np.zeros((image_side_length, image_side_length, 3))

        history = self.track_helper.get_past_for_sample(sample_token,
                                                  self.seconds_of_history,
                                                  in_agent_frame=False,
                                                  just_xy=False)
        history = reverse_history(history)
        #print(history.keys())

        present_time = self.track_helper.get_annotations_for_sample(sample_token)

        history = self.add_present_time_to_history(present_time, history)

        # gt_history = self.nusc_helper.get_past_for_sample(sample_token,
        #                                           self.seconds_of_history,
        #                                           in_agent_frame=False,
        #                                           just_xy=False)
        # gt_history[instance_token].reverse()
        # history[instance_token] = gt_history[instance_token]

        #center_agent_annotation = self.nusc_helper.get_sample_annotation(instance_token, sample_token)
        center_agent_annotation = self.track_helper.get_sample_annotation(instance_token, sample_token)

        self.draw_agent_boxes(center_agent_annotation, central_track_pixels,
                         history, base_image, resolution=self.resolution, get_color=self.color_map)

        center_agent_yaw = quaternion_yaw(Quaternion(center_agent_annotation['rotation']))
        rotation_mat = get_rotation_matrix(base_image.shape, center_agent_yaw)

        rotated_image = cv2.warpAffine(base_image, rotation_mat, (base_image.shape[1],
                                                                  base_image.shape[0]))

        row_crop, col_crop = get_crops(self.meters_ahead, self.meters_behind,
                                       self.meters_left, self.meters_right, self.resolution,
                                       image_side_length)

        return rotated_image[row_crop, col_crop].astype('uint8')



class StaticLayerFromTracking(StaticLayerRepresentation):
    """
    Creates a representation of the static map layers where
    the map layers are given a color and rasterized onto a
    three channel image.
    """

    def __init__(self, tracking_helper: TrackingResultsPredictHelper,
                 layer_names: List[str] = None,
                 colors: List[Color] = None,
                 resolution: float = 0.1, # meters / pixel
                 meters_ahead: float = 40, meters_behind: float = 10,
                 meters_left: float = 25, meters_right: float = 25):

        self.track_helper = tracking_helper
        self.nusc_helper = PredictHelper(tracking_helper.data)
        self.maps = load_all_maps(self.nusc_helper)

        if not layer_names:
            layer_names = ['drivable_area', 'ped_crossing', 'walkway']
        self.layer_names = layer_names

        if not colors:
            colors = [(255, 255, 255), (119, 136, 153), (0, 0, 255)]
        self.colors = colors

        self.resolution = resolution
        self.meters_ahead = meters_ahead
        self.meters_behind = meters_behind
        self.meters_left = meters_left
        self.meters_right = meters_right
        self.combinator = Rasterizer()

    def make_representation(self, instance_token: str, sample_token: str) -> np.ndarray:
        """
        Makes rasterized representation of static map layers.
        :param instance_token: Token for instance.
        :param sample_token: Token for sample.
        :return: Three channel image.
        """

        #sample_annotation = self.helper.get_sample_annotation(instance_token, sample_token)
        sample_annotation = self.track_helper.get_sample_annotation(instance_token, sample_token)
        map_name = self.nusc_helper.get_map_name_from_sample_token(sample_token)

        x, y = sample_annotation['translation'][:2]

        yaw = quaternion_yaw(Quaternion(sample_annotation['rotation']))

        yaw_corrected = correct_yaw(yaw)

        image_side_length = 2 * max(self.meters_ahead, self.meters_behind,
                                    self.meters_left, self.meters_right)
        image_side_length_pixels = int(image_side_length / self.resolution)

        patchbox = get_patchbox(x, y, image_side_length)

        angle_in_degrees = angle_of_rotation(yaw_corrected) * 180 / np.pi

        canvas_size = (image_side_length_pixels, image_side_length_pixels)

        masks = self.maps[map_name].get_map_mask(patchbox, angle_in_degrees, self.layer_names, canvas_size=canvas_size)

        images = []
        for mask, color in zip(masks, self.colors):
            images.append(change_color_of_binary_mask(np.repeat(mask[::-1, :, np.newaxis], 3, 2), color))

        lanes = draw_lanes_in_agent_frame(image_side_length_pixels, x, y, yaw, radius=50,
                                          image_resolution=self.resolution, discretization_resolution_meters=1,
                                          map_api=self.maps[map_name])

        images.append(lanes)

        image = self.combinator.combine(images)

        row_crop, col_crop = get_crops(self.meters_ahead, self.meters_behind, self.meters_left,
                                       self.meters_right, self.resolution,
                                       int(image_side_length / self.resolution))

        return image[row_crop, col_crop, :]