import colorsys
from typing import Any, Dict, List, Tuple, Callable

import cv2
import numpy as np
from pyquaternion import Quaternion

from nuscenes.prediction import PredictHelper
from nuscenes.prediction.helper import quaternion_yaw
from nuscenes.prediction.input_representation.utils import convert_to_pixel_coords, get_crops, get_rotation_matrix

History = Dict[str, List[Dict[str, Any]]]


def fade_color(color: Tuple[int, int, int],
               step: int,
               total_number_of_steps: int) -> Tuple[int, int, int]:
    """
    Fades a color so that past observations are darker in the image.
    :param color: Tuple of ints describing an RGB color.
    :param step: The current time step.
    :param total_number_of_steps: The total number of time steps
        the agent has in the image.
    :return: Tuple representing faded rgb color.
    """

    LOWEST_VALUE = 0.4

    if step == total_number_of_steps:
        return color

    hsv_color = colorsys.rgb_to_hsv(*color)

    increment = (float(hsv_color[2])/255. - LOWEST_VALUE) / total_number_of_steps

    new_value = LOWEST_VALUE + step * increment

    new_rgb = colorsys.hsv_to_rgb(float(hsv_color[0]),
                                  float(hsv_color[1]),
                                  new_value * 255.)
    return new_rgb


def default_colors(category: int) -> Tuple[int, int, int]:
    """
    Maps a category name to an rgb color (without fading).
    :param category_name: Name of object category for the annotation.
    :return: Tuple representing rgb color.
    """

    if category == 1:
        return 255, 255, 0  # yellow
    elif category == 2:
        return 204, 0, 204  # violet
    elif category == 3:
        return 255, 153, 51  # orange
    else:
        raise ValueError(f"Cannot map {category_name} to a color.")


def draw_agent_boxes(center_agent_annotation: Dict[str, Any],
                     center_agent_pixels: Tuple[float, float],
                     preds: Dict[str, Any],
                     base_image: np.ndarray,
                     get_color: Callable[[str], Tuple[int, int, int]],
                     resolution: float = 0.1) -> None:
    """
    Draws past sequence of agent boxes on the image.
    (param) center_agent_annotation: not being used
    (param) center_agent_pixels: Pixel location of the agent in center of the image.
    (param) agent_history: History for all agents in the scene.
    (param) base_image: Image to draw the agents in.
    (param) get_color: Mapping from object index to RGB tuple. No longer used
    (param) resolution: Size of the image in pixels / meter.
    returns None.
    """

    agent_x, agent_y = 0,0 
    for i, box in enumerate(preds['pred_boxes']):
        x, y, z, dx, dy, dz, yaw_in_radians = box

        column_pixel = int(-y / .1 + center_agent_pixels[1])
        row_pixel = int(-x / .1 + center_agent_pixels[0])
        width_in_pixels = int(dx /.1)
        length_in_pixels = int(dy /.1)

        coord_tuple = ((column_pixel, row_pixel), (length_in_pixels, width_in_pixels), -yaw_in_radians * 180 / np.pi)

        boxPoints = cv2.boxPoints(coord_tuple)
        color = default_colors(agent_t.category)

        # Don't fade the colors if there is no history
        #if num_points > 1:
        #    color = fade_color(color, i, num_points - 1)

        cv2.fillPoly(base_image, pts=[np.int0(boxPoints)], color=color)

def draw_agent_from_history(center_agent_annotation: Dict[str, Any],
                     center_agent_pixels: Tuple[float, float],
                     trackingInfo,
                     currentTime,
                     base_image: np.ndarray,
                     get_color: Callable[[str], Tuple[int, int, int]],
                     resolution: float = 0.1) -> None:
    """
    Draws past sequence of agent boxes on the image.
    (param) center_agent_annotation: No longer being used
    (param) center_agent_pixels: Pixel location of the agent in center of the image.
    (param) agent_history: History for all agents in the scene.
    (param) base_image: Image to draw the agents in.
    (param) get_color: Mapping from category_name to RGB tuple.
    (param) resolution: Size of the image in pixels / meter.
    returns None.
    """
    history = trackingInfo["history"]
    offsets = trackingInfo["gps"]
    heading = trackingInfo["heading"]

    #Draw the ego vehicle since it is not detected in the PV-RCNN
    agent_x, agent_y = 0,0
    for i,(offset_x, offset_y,offset_yaw) in enumerate(offsets):
        #only draw the last 3 timestep at 2Hz 
        timeIndex = (currentTime - i)//5
        if i % 5 != 0 or not timeIndex in [2, 1, 0]:
            continue
        #take the current agent data in local frame and put it into global frame 
        alignment = -heading

        translation_rotation = np.array([[np.cos(alignment), -np.sin(alignment)], [np.sin(alignment), np.cos(alignment)]])
        rotation = np.array([[np.cos(offset_yaw), -np.sin(offset_yaw)], [np.sin(offset_yaw), np.cos(offset_yaw)]])
        offset_x, offset_y = translation_rotation @ np.array([offset_x, offset_y])

        column_pixel = int((-offset_y)/ .1 + center_agent_pixels[1])
        row_pixel = int((-offset_x)/ .1 + center_agent_pixels[0])
        coord_tuple = ((column_pixel, row_pixel), (10, 30), (-offset_yaw) * 180 / np.pi)
        color = (255,0,0)
        color = fade_color((255,0,0), 2 - timeIndex, 2)
        boxPoints = cv2.boxPoints(coord_tuple)
        cv2.fillPoly(base_image, pts=[np.int0(boxPoints)], color=color)

    #Draw all other agents in the scene
    for i, agentHistory in enumerate(history):
        for step, agent_t in enumerate(agentHistory):
            #only draw the last 3 timestep at 2Hz 
            timeIndex = (currentTime - agent_t.timeStep)//5
            if not timeIndex in [2, 1, 0]:
                continue
            #take the current agent data in local frame and put it into global frame 
            x, y, z, dx, dy, dz, yaw_in_radians = agent_t.boxDims
            
            offset_x, offset_y, offset_yaw = offsets[agent_t.timeStep]
            alignment = -heading

            translation_rotation = np.array([[np.cos(alignment), -np.sin(alignment)], [np.sin(alignment), np.cos(alignment)]])
            rotation = np.array([[np.cos(offset_yaw), -np.sin(offset_yaw)], [np.sin(offset_yaw), np.cos(offset_yaw)]])
            x, y = rotation @ np.array([x,y])
            offset_x, offset_y = translation_rotation @ np.array([offset_x, offset_y])

            column_pixel = int((-y - offset_y)/ .1 + center_agent_pixels[1])
            row_pixel = int((-x - offset_x)/ .1 + center_agent_pixels[0])
            width_in_pixels = int(dx /.1)
            length_in_pixels = int(dy /.1)

            coord_tuple = ((column_pixel, row_pixel), (length_in_pixels, width_in_pixels), (-offset_yaw-yaw_in_radians) * 180 / np.pi)

            boxPoints = cv2.boxPoints(coord_tuple)
            
            color = agent_t.color
            color = fade_color(color, 2 - timeIndex, 2)

            cv2.fillPoly(base_image, pts=[np.int0(boxPoints)], color=color)

class AgentBox:
    """
    Represents the past sequence of agent states as a three-channel
    image with faded 2d boxes.
    """

    def __init__(self,
                 seconds_of_history: float = 2,
                 frequency_in_hz: float = 2,
                 resolution: float = 0.1,  # meters / pixel
                 meters_ahead: float = 40, meters_behind: float = 10,
                 meters_left: float = 25, meters_right: float = 25,
                 color_mapping: Callable[[str], Tuple[int, int, int]] = None):

        self.seconds_of_history = seconds_of_history
        self.frequency_in_hz = frequency_in_hz

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

    def getBoxes(self, preds, colorFunc=None, agentHistory=None, currentTime=None) -> np.ndarray:
        """
        Draws agent boxes with faded history into a black background.
        (param) preds:        dictionary produced by object detection (PV-RCNN) containing boundind box data etc.
        (param) colorFunc:    function takes in object index and returns color (used for testing)
        (param) agentHistory: dictionary from tracker containing tracked objects
        (param) currentTiem:  int timestep 
        returs np.ndarray representing a 3 channel image.
        """
        if not colorFunc:
            colorFunc = lambda i: default_colors(preds['pred_labels'][i])

        # Taking radius around track before to ensure all actors are in image
        buffer = max([self.meters_ahead, self.meters_behind,
                      self.meters_left, self.meters_right]) * 2

        image_side_length = int(buffer/self.resolution)

        # We will center the track in the image
        central_track_pixels = (3 * image_side_length / 4, image_side_length / 2)

        base_image = np.zeros((image_side_length, image_side_length, 3))
        if agentHistory:
            draw_agent_from_history({}, central_track_pixels,
                 agentHistory, currentTime, base_image, resolution=self.resolution, get_color=colorFunc)
        else:
            draw_agent_boxes({}, central_track_pixels,
                 preds, base_image, resolution=self.resolution, get_color=colorFunc)

        return base_image.astype('uint8')

    def make_representation(self, instance_token: str, sample_token: str) -> np.ndarray:
        """
        draws a specific scene of the KITTI dataset.
        """
        preds = {'pred_boxes': np.array([[ 14.7293,  -1.0697,  -0.7887,   3.7134,   1.5743,   1.4734,   5.9638],
        [  8.1551,   1.2285,  -0.8339,   3.6470,   1.5568,   1.5720,   2.8300],
        [  3.9858,   2.6945,  -0.8672,   3.3475,   1.6138,   1.5873,   6.0024],
        [  6.5308,  -3.8682,  -1.0617,   2.9842,   1.3960,   1.4679,   5.9861],
        [ 40.9299,  -9.8144,  -0.5646,   3.8337,   1.6143,   1.5887,   5.9492],
        [ 33.5855,  -7.1258,  -0.4450,   4.1105,   1.7229,   1.7785,   2.8397],
        [ 20.2270,  -8.4760,  -0.9010,   2.5252,   1.5289,   1.5415,   5.9202],
        [ 24.9747, -10.3103,  -0.9713,   3.9088,   1.6942,   1.4646,   5.8124],
        [ 55.6275, -20.2395,  -0.5136,   4.3099,   1.7079,   1.6050,   2.8220],
        [ 28.8354,  -1.3581,  -0.3444,   4.0354,   1.5510,   1.5145,   4.3867],
        [ 33.5991, -15.3848,  -0.5443,   1.7486,   0.5379,   1.7414,   2.9633],
        [ 37.1934,  -6.1262,  -0.3610,   0.6718,   0.6549,   1.8252,   6.2896],
        [ 34.1049,  -4.9239,  -0.3691,   0.6049,   0.6690,   1.8181,   6.1904],
        [ 30.3460,  -3.7285,  -0.4023,   1.8521,   0.5506,   1.7339,   6.1950],
        [  2.5116,   6.4730,  -0.8472,   4.1226,   1.5696,   1.4396,   2.7402],
        [ 29.8615, -14.1368,  -0.7158,   0.8671,   0.6614,   1.8148,   2.6203],
        [  0.3147,  -0.1887,  -0.9616,   3.6420,   1.5767,   1.5107,   6.1641],
        [ 40.5936,  -7.1140,  -0.4020,   0.6655,   0.6505,   1.8205,   6.2723]]), 
        'pred_scores': np.array([0.9998, 0.9998, 0.9980, 0.9968, 0.9932, 0.9905, 0.9633, 0.9394, 0.8998,
        0.8671, 0.7343, 0.5371, 0.5337, 0.4154, 0.2899, 0.2844, 0.1963, 0.1569]),
        'pred_labels': np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 2, 2, 3, 1, 2, 1, 2])}
        preds2 = {key:val[16:17] for key, val in preds.items()}
        return self.getBoxes(preds)
