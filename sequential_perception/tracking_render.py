import os
import string
from typing import Any, List, Tuple
from collections import defaultdict
import copy

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from pandas import DataFrame
from pyquaternion import Quaternion

from nuscenes.eval.common.render import setup_axis
from nuscenes.eval.tracking.constants import TRACKING_COLORS, PRETTY_TRACKING_NAMES
from nuscenes.eval.tracking.data_classes import TrackingBox, TrackingMetricDataList
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points, transform_matrix


class RenderBox:
    """ Simple data class representing a 3d box including, label, score and velocity. """

    def __init__(self,
                 center: List[float],
                 size: List[float],
                 orientation: Quaternion,
                 label: int = np.nan,
                 score: float = np.nan,
                 velocity: Tuple = (np.nan, np.nan, np.nan),
                 name: str = None,
                 token: str = None):
        """
        :param center: Center of box given as x, y, z.
        :param size: Size of box in width, length, height.
        :param orientation: Box orientation.
        :param label: Integer label, optional.
        :param score: Classification score, optional.
        :param velocity: Box velocity in x, y, z direction.
        :param name: Box name, optional. Can be used e.g. for denote category name.
        :param token: Unique string identifier from DB.
        """
        assert not np.any(np.isnan(center))
        assert not np.any(np.isnan(size))
        assert len(center) == 3
        assert len(size) == 3
        assert type(orientation) == Quaternion

        self.center = np.array(center)
        self.wlh = np.array(size)
        self.orientation = orientation
        self.label = int(label) if not np.isnan(label) else label
        self.score = float(score) if not np.isnan(score) else score
        self.velocity = np.array(velocity)
        self.name = name
        self.token = token

    def __eq__(self, other):
        center = np.allclose(self.center, other.center)
        wlh = np.allclose(self.wlh, other.wlh)
        orientation = np.allclose(self.orientation.elements, other.orientation.elements)
        label = (self.label == other.label) or (np.isnan(self.label) and np.isnan(other.label))
        score = (self.score == other.score) or (np.isnan(self.score) and np.isnan(other.score))
        vel = (np.allclose(self.velocity, other.velocity) or
               (np.all(np.isnan(self.velocity)) and np.all(np.isnan(other.velocity))))

        return center and wlh and orientation and label and score and vel

    def __repr__(self):
        repr_str = 'label: {}, score: {:.2f}, xyz: [{:.2f}, {:.2f}, {:.2f}], wlh: [{:.2f}, {:.2f}, {:.2f}], ' \
                   'rot axis: [{:.2f}, {:.2f}, {:.2f}], ang(degrees): {:.2f}, ang(rad): {:.2f}, ' \
                   'vel: {:.2f}, {:.2f}, {:.2f}, name: {}, token: {}'

        return repr_str.format(self.label, self.score, self.center[0], self.center[1], self.center[2], self.wlh[0],
                               self.wlh[1], self.wlh[2], self.orientation.axis[0], self.orientation.axis[1],
                               self.orientation.axis[2], self.orientation.degrees, self.orientation.radians,
                               self.velocity[0], self.velocity[1], self.velocity[2], self.name, self.token)

    @property
    def rotation_matrix(self) -> np.ndarray:
        """
        Return a rotation matrix.
        :return: <np.float: 3, 3>. The box's rotation matrix.
        """
        return self.orientation.rotation_matrix

    def translate(self, x: np.ndarray) -> None:
        """
        Applies a translation.
        :param x: <np.float: 3, 1>. Translation in x, y, z direction.
        """
        self.center += x

    def rotate(self, quaternion: Quaternion) -> None:
        """
        Rotates box.
        :param quaternion: Rotation to apply.
        """
        self.center = np.dot(quaternion.rotation_matrix, self.center)
        self.orientation = quaternion * self.orientation
        self.velocity = np.dot(quaternion.rotation_matrix, self.velocity)

    def corners(self, wlh_factor: float = 1.0) -> np.ndarray:
        """
        Returns the bounding box corners.
        :param wlh_factor: Multiply w, l, h by a factor to scale the box.
        :return: <np.float: 3, 8>. First four corners are the ones facing forward.
            The last four are the ones facing backwards.
        """
        w, l, h = self.wlh * wlh_factor

        # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
        x_corners = l / 2 * np.array([1,  1,  1,  1, -1, -1, -1, -1])
        y_corners = w / 2 * np.array([1, -1, -1,  1,  1, -1, -1,  1])
        z_corners = h / 2 * np.array([1,  1, -1, -1,  1,  1, -1, -1])
        corners = np.vstack((x_corners, y_corners, z_corners))

        # Rotate
        corners = np.dot(self.orientation.rotation_matrix, corners)

        # Translate
        x, y, z = self.center
        corners[0, :] = corners[0, :] + x
        corners[1, :] = corners[1, :] + y
        corners[2, :] = corners[2, :] + z

        return corners

    def bottom_corners(self) -> np.ndarray:
        """
        Returns the four bottom corners.
        :return: <np.float: 3, 4>. Bottom corners. First two face forward, last two face backwards.
        """
        return self.corners()[:, [2, 3, 7, 6]]

    def render(self,
               axis: Axes,
               view: np.ndarray = np.eye(3),
               normalize: bool = False,
               colors: Tuple = ('b', 'r', 'k'),
               linewidth: float = 2,
               alpha: float = 1) -> None:
        """
        Renders the box in the provided Matplotlib axis.
        :param axis: Axis onto which the box should be drawn.
        :param view: <np.array: 3, 3>. Define a projection in needed (e.g. for drawing projection in an image).
        :param normalize: Whether to normalize the remaining coordinate.
        :param colors: (<Matplotlib.colors>: 3). Valid Matplotlib colors (<str> or normalized RGB tuple) for front,
            back and sides.
        :param linewidth: Width in pixel of the box sides.
        """
        corners = view_points(self.corners(), view, normalize=normalize)[:2, :]

        def draw_rect(selected_corners, color, alpha):
            prev = selected_corners[-1]
            for corner in selected_corners:
                axis.plot([prev[0], corner[0]], [prev[1], corner[1]], color=color, linewidth=linewidth, alpha=alpha)
                prev = corner

        # Draw the sides
        for i in range(4):
            axis.plot([corners.T[i][0], corners.T[i + 4][0]],
                      [corners.T[i][1], corners.T[i + 4][1]],
                      color=colors[2], linewidth=linewidth, alpha=alpha)

        # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
        draw_rect(corners.T[:4], colors[0], alpha)
        draw_rect(corners.T[4:], colors[1], alpha)

        # Draw line indicating the front
        center_bottom_forward = np.mean(corners.T[2:4], axis=0)
        center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
        axis.plot([center_bottom[0], center_bottom_forward[0]],
                  [center_bottom[1], center_bottom_forward[1]],
                  color=colors[0], linewidth=linewidth, alpha=alpha)

    def copy(self) -> 'Box':
        """
        Create a copy of self.
        :return: A copy.
        """
        return copy.deepcopy(self)



class CustomTrackingRenderer:
    """
    Class that renders the tracking results in BEV and saves them to a folder.
    """
    def __init__(self, save_path):
        """
        :param save_path:  Output path to save the renderings.
        """
        self.save_path = save_path
        self.id2color = {}  # The color of each track.

    def render_bev(self, timestamp: int, frame_gt: List[TrackingBox], frame_pred: List[TrackingBox]) \
            -> None:
        """
        Render function for a given scene timestamp
        :param events: motmetrics events for that particular
        :param timestamp: timestamp for the rendering
        :param frame_gt: list of ground truth boxes
        :param frame_pred: list of prediction boxes
        """
        # Init.
        print('Rendering {}'.format(timestamp))
        #switches = events[events.Type == 'SWITCH']
        #switch_ids = switches.HId.values
        fig, ax = plt.subplots()

        # Plot GT boxes.
        for b in frame_gt:
            color = 'k'
            box = Box(b.ego_translation, b.size, Quaternion(b.rotation), name=b.tracking_name, token=b.tracking_id)
            box.render(ax, view=np.eye(4), colors=(color, color, color), linewidth=1)

        # Plot predicted boxes.
        for b in frame_pred:
            box = Box(b.ego_translation, b.size, Quaternion(b.rotation), name=b.tracking_name, token=b.tracking_id)

            # Determine color for this tracking id.
            if b.tracking_id not in self.id2color.keys():
                self.id2color[b.tracking_id] = (float(hash(b.tracking_id + 'r') % 256) / 255,
                                                float(hash(b.tracking_id + 'g') % 256) / 255,
                                                float(hash(b.tracking_id + 'b') % 256) / 255)

            # Render box. Highlight identity switches in red.
            # if b.tracking_id in switch_ids:
            #     color = self.id2color[b.tracking_id]
            #     box.render(ax, view=np.eye(4), colors=('r', 'r', color))
            # else:
            color = self.id2color[b.tracking_id]
            box.render(ax, view=np.eye(4), colors=(color, color, color))
            plt.text(b.ego_translation[0], b.ego_translation[1], s=str(b.tracking_id).upper(), fontsize='xx-small')

        # Plot ego pose.
        plt.scatter(0, 0, s=96, facecolors='none', edgecolors='k', marker='o')
        plt.xlim(-50, 50)
        plt.ylim(-50, 50)

        # Save to disk and close figure.
        fig.savefig(os.path.join(self.save_path, 'custom_local_{}.png'.format(timestamp)))
        plt.close(fig)

    # def render_global(self,  frame_gt: List[TrackingBox], frame_pred: List[TrackingBox], tracking_id:List[]) \
    #         -> None:

    def render_tracks_in_global(self, tracks, render_ids):
        """
        Render function for a given scene timestamp
        :param events: motmetrics events for that particular
        :param timestamp: timestamp for the rendering
        :param frame_gt: list of ground truth boxes
        :param frame_pred: list of prediction boxes
        """
        # Init.
        #print('Rendering {}'.format(timestamp))
        #switches = events[events.Type == 'SWITCH']
        #switch_ids = switches.HId.values
        fig, ax = plt.subplots()

        # Plot GT boxes.
        # for b in frame_gt:
        #     color = 'k'
        #     box = Box(b.ego_translation, b.size, Quaternion(b.rotation), name=b.tracking_name, token=b.tracking_id)
        #     box.render(ax, view=np.eye(4), colors=(color, color, color), linewidth=1)

        for tstamp in tracks.keys():
        # Plot predicted boxes.
            frame_pred = [tr for tr in tracks[tstamp]]
            for b in frame_pred:
                print(b.tracking_id)
                if b.tracking_id in render_ids:
                    print('HIT')
                    print(b.tracking_id)
                    box = Box(b.translation, b.size, Quaternion(b.rotation), name=b.tracking_name, token=b.tracking_id)

                    # Determine color for this tracking id.
                    if b.tracking_id not in self.id2color.keys():
                        self.id2color[b.tracking_id] = (float(hash(b.tracking_id + 'r') % 256) / 255,
                                                        float(hash(b.tracking_id + 'g') % 256) / 255,
                                                        float(hash(b.tracking_id + 'b') % 256) / 255)
                    
                    color = self.id2color[b.tracking_id]
                    box.render(ax, view=np.eye(4), colors=(color, color, color))



        # Save to disk and close figure.
        fig.savefig(os.path.join(self.save_path, 'custom_local_{}.png'.format(3)))
        plt.close(fig)


    def render_fade_histories(self, tracks_gt, tracks_pred, ids_gt, ids_pred):
        """
        Render function for a given scene timestamp
        :param events: motmetrics events for that particular
        :param timestamp: timestamp for the rendering
        :param frame_gt: list of ground truth boxes
        :param frame_pred: list of prediction boxes
        """
        # Init.
        #print('Rendering {}'.format(timestamp))
        #switches = events[events.Type == 'SWITCH']
        #switch_ids = switches.HId.values
        

        # Plot GT boxes.
        # for b in frame_gt:
        #     color = 'k'
        #     box = Box(b.ego_translation, b.size, Quaternion(b.rotation), name=b.tracking_name, token=b.tracking_id)
        #     box.render(ax, view=np.eye(4), colors=(color, color, color), linewidth=1)
        
        pred_boxes = defaultdict(lambda : [])
        true_boxes = defaultdict(lambda : [])
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:pink', 'tab:gray']

        for tstamp in tracks_pred.keys():
        # Plot predicted boxes.
            frame_pred = [tr for tr in tracks_pred[tstamp]]
            for b in frame_pred:
                if b.tracking_id in ids_pred:
                    box = RenderBox(b.translation, b.size, Quaternion(b.rotation), name=b.tracking_name, token=b.tracking_id)
                    pred_boxes[b.tracking_id].append(box)

                    # Determine color for this tracking id.
                    if b.tracking_id not in self.id2color.keys():
                        # self.id2color[b.tracking_id] = (float(hash(b.tracking_id + 'r') % 256) / 255,
                        #                                 float(hash(b.tracking_id + 'g') % 256) / 255,
                        #                                 float(hash(b.tracking_id + 'b') % 256) / 255)
                        self.id2color[b.tracking_id] = colors[ids_pred.index(b.tracking_id)]
                

                    frame_true = [tr for tr in tracks_gt[tstamp]]
                    for bgt in frame_true:
                        #if bgt.tracking_id in ids_gt:
                        if bgt.tracking_id ==  ids_gt[ids_pred.index(b.tracking_id)]:
                            box = RenderBox(bgt.translation, bgt.size, Quaternion(bgt.rotation), name=bgt.tracking_name, token=bgt.tracking_id)
                            true_boxes[bgt.tracking_id].append(box)
                    
                    #color = self.id2color[b.tracking_id]
                    #box.render(ax, view=np.eye(4), colors=(color, color, color))
        fig, ax = plt.subplots()
        for id in pred_boxes.keys():
            color = self.id2color[id]
            alphas = np.linspace(0.05, 1.0, num=len(pred_boxes[id]))
            #alphas = np.logspace(0.1, 1.0, num=len(pred_boxes[id]))/10
            print(alphas)
            for idx, b in enumerate(pred_boxes[id]):
                b.render(ax, view=np.eye(4), colors=(color, color, color), alpha=alphas[idx])

        # Save to disk and close figure.
        fig.savefig(os.path.join(self.save_path, 'custom_local_{}_faded.png'.format(1)))
        plt.close(fig)

        fig, ax = plt.subplots()
        for id in true_boxes.keys():
            pid = ids_pred[ids_gt.index(id)]
            color = self.id2color[pid]
            alphas = np.linspace(0.05, 1.0, num=len(true_boxes[id]))
            #alphas = np.logspace(0.1, 1.0, num=len(true_boxes[id]))/10
            #alphas = np.linspace(0.2, 1.0, num=len(true_boxes[id]))**2
            #print(alphas)
            for idx, b in enumerate(true_boxes[id]):
                b.render(ax, view=np.eye(4), colors=(color, color, color), alpha=alphas[idx])

        # Save to disk and close figure.
        fig.savefig(os.path.join(self.save_path, 'custom_local_{}_faded_gt.png'.format(1)))
        plt.close(fig)