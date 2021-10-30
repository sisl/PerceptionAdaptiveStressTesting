from copy import deepcopy
import os.path as osp
import time
from typing import Tuple, List

import numpy as np
from pyquaternion import Quaternion

from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset

from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from nuscenes.utils.color_map import get_colormap



class PCDetNuScenesDataset:
    def __init__(self, nuscenes, pcdet_infos):
        self.nuscenes = nuscenes
        self.pcdet_infos = pcdet_infos
        self.token_to_idx_map = self._get_reverse_sample_map()
        return

    def _get_reverse_sample_map(self):
        reverse_map = {}
        for idx, data_dict in enumerate(self.pcdet_infos):
            sample_token = data_dict['metadata']['token']
            reverse_map[sample_token] = idx
        
        return reverse_map

    def get_data_for_sample(self, sample_token: str):
        idx = self.token_to_idx_map[sample_token]
        data_dict = self.pcdet_dataset[idx]
        return data_dict



class NuScenesScene:
    '''NuScenes object for a single scene'''
    def __init__(self, dataroot:str, nusc:NuScenes, scene_token:str):
        self.version = nusc.version
        self.dataroot = dataroot
        self.table_names = ['category', 'attribute', 'visibility', 'instance', 'sensor', 'calibrated_sensor',
                            'ego_pose', 'log', 'scene', 'sample', 'sample_data', 'sample_annotation', 'map']


        # Build new tables
        self.category = deepcopy(nusc.category)
        self.attribute = deepcopy(nusc.attribute)
        self.visibility = deepcopy(nusc.visibility)
        #self.instance = self.__load_table__('instance')
        self.sensor = deepcopy(nusc.sensor)
        #self.calibrated_sensor = nusc.calibrated_sensor
        #self.ego_pose = self.__load_table__('ego_pose')
        #self.log = self.__load_table__('log')
        #self.scene = self.__load_table__('scene')
        #self.sample = self.__load_table__('sample')
        #self.sample_data = self.__load_table__('sample_data')
        #self.sample_annotation = self.__load_table__('sample_annotation')
        self.map = deepcopy(nusc.map)

        # Get all sample tokens for the scene
        first_sample_token = nusc.get('scene', scene_token)['first_sample_token']
        scene_sample_tokens = []
        cur_sample_token = first_sample_token
        while cur_sample_token != '':
            scene_sample_tokens.append(cur_sample_token)
            cur_sample_token = nusc.get('sample', cur_sample_token)['next']

        #print(scene_sample_tokens)

        # make new scene table
        self.scene = [deepcopy(nusc.get('scene', scene_token))]

        # make new log table
        self.log = [deepcopy(nusc.get('log', self.scene[0]['log_token']))]


        sensor_keys = list(nusc.get('sample', first_sample_token)['data'].keys())

        # make new everything else
        self.ego_pose = []
        self.calibrated_sensor =[]
        self.sample = []
        self.instance = []
        self.sample_data = []
        self.sample_annotation = []

        instance_tokens = []
        ego_tokens = []
        cal_sensor_tokens = []

        for sample_token in scene_sample_tokens:

            sample = deepcopy(nusc.get('sample', sample_token)) # gets a list of annotation tokens AND data tokens
            
            self.sample.append(sample)

            ann_tokens = sample['anns']

            for ann_token in ann_tokens:
                ann = deepcopy(nusc.get('sample_annotation', ann_token))
                self.sample_annotation.append(ann)

                if ann['instance_token'] not in instance_tokens:
                    instance_tokens.append(ann['instance_token'])
                    inst = deepcopy(nusc.get('instance', ann['instance_token']))
                    self.instance.append(inst)



            # For each sensor
            for sensor_key in sensor_keys:
                sensor_data_token = sample['data'][sensor_key]

                # add sensor_data record
                data_record = deepcopy(nusc.get('sample_data', sensor_data_token))
                self.sample_data.append(data_record)

                # sample_data_record points to ego_pose and calibrated_sensor
                # ego_pose
                if data_record['ego_pose_token'] not in ego_tokens:
                    ego_tokens.append(data_record['ego_pose_token'])
                    pose_record = deepcopy(nusc.get('ego_pose', data_record['ego_pose_token']))
                    self.ego_pose.append(pose_record)

                # calibrated_sensor
                if data_record['calibrated_sensor_token'] not in cal_sensor_tokens:
                    cal_sensor_tokens.append(data_record['calibrated_sensor_token'])
                    cal_sensor_record = deepcopy(nusc.get('calibrated_sensor', data_record['calibrated_sensor_token']))
                    self.calibrated_sensor.append(cal_sensor_record)


        self.colormap = get_colormap()


        self.__make_reverse_index__(nusc.verbose)

    # def __init__(self, dataroot:str, nusc:NuScenes, scene_token:str):
    #     self.version = nusc.version
    #     self.dataroot = dataroot
    #     self.table_names = ['category', 'attribute', 'visibility', 'instance', 'sensor', 'calibrated_sensor',
    #                         'ego_pose', 'log', 'scene', 'sample', 'sample_data', 'sample_annotation', 'map']


    #     # Build new tables
    #     self.category = nusc.category
    #     self.attribute = nusc.attribute
    #     self.visibility = nusc.visibility
    #     #self.instance = self.__load_table__('instance')
    #     self.sensor = nusc.sensor
    #     #self.calibrated_sensor = nusc.calibrated_sensor
    #     #self.ego_pose = self.__load_table__('ego_pose')
    #     #self.log = self.__load_table__('log')
    #     #self.scene = self.__load_table__('scene')
    #     #self.sample = self.__load_table__('sample')
    #     #self.sample_data = self.__load_table__('sample_data')
    #     #self.sample_annotation = self.__load_table__('sample_annotation')
    #     self.map = nusc.map

    #     # Get all sample tokens for the scene
    #     first_sample_token = nusc.get('scene', scene_token)['first_sample_token']
    #     scene_sample_tokens = []
    #     cur_sample_token = first_sample_token
    #     while cur_sample_token != '':
    #         scene_sample_tokens.append(cur_sample_token)
    #         cur_sample_token = nusc.get('sample', cur_sample_token)['next']

    #     #print(scene_sample_tokens)

    #     # make new scene table
    #     self.scene = [nusc.get('scene', scene_token)]

    #     # make new log table
    #     self.log = [nusc.get('log', self.scene[0]['log_token'])]


    #     sensor_keys = list(nusc.get('sample', first_sample_token)['data'].keys())

    #     # make new everything else
    #     self.ego_pose = []
    #     self.calibrated_sensor =[]
    #     self.sample = []
    #     self.instance = []
    #     self.sample_data = []
    #     self.sample_annotation = []

    #     instance_tokens = []
    #     ego_tokens = []
    #     cal_sensor_tokens = []

    #     for sample_token in scene_sample_tokens:

    #         sample = nusc.get('sample', sample_token) # gets a list of annotation tokens AND data tokens
            
    #         self.sample.append(sample)

    #         ann_tokens = sample['anns']

    #         for ann_token in ann_tokens:
    #             ann = nusc.get('sample_annotation', ann_token)
    #             self.sample_annotation.append(ann)

    #             if ann['instance_token'] not in instance_tokens:
    #                 instance_tokens.append(ann['instance_token'])
    #                 inst = nusc.get('instance', ann['instance_token'])
    #                 self.instance.append(inst)



    #         # For each sensor
    #         for sensor_key in sensor_keys:
    #             sensor_data_token = sample['data'][sensor_key]

    #             # add sensor_data record
    #             data_record = nusc.get('sample_data', sensor_data_token)
    #             self.sample_data.append(data_record)

    #             # sample_data_record points to ego_pose and calibrated_sensor
    #             # ego_pose
    #             if data_record['ego_pose_token'] not in ego_tokens:
    #                 ego_tokens.append(data_record['ego_pose_token'])
    #                 pose_record = nusc.get('ego_pose', data_record['ego_pose_token'])
    #                 self.ego_pose.append(pose_record)

    #             # calibrated_sensor
    #             if data_record['calibrated_sensor_token'] not in cal_sensor_tokens:
    #                 cal_sensor_tokens.append(data_record['calibrated_sensor_token'])
    #                 cal_sensor_record = nusc.get('calibrated_sensor', data_record['calibrated_sensor_token'])
    #                 self.calibrated_sensor.append(cal_sensor_record)


    #     self.colormap = get_colormap()


    #     self.__make_reverse_index__(nusc.verbose)
            

    def __make_reverse_index__(self, verbose: bool) -> None:
        """
        De-normalizes database to create reverse indices for common cases.
        :param verbose: Whether to print outputs.
        """

        start_time = time.time()
        if verbose:
            print("Reverse indexing ...")

        # Store the mapping from token to table index for each table.
        self._token2ind = dict()
        for table in self.table_names:
            self._token2ind[table] = dict()

            for ind, member in enumerate(getattr(self, table)):
                self._token2ind[table][member['token']] = ind

        # Decorate (adds short-cut) sample_annotation table with for category name.
        for record in self.sample_annotation:
            inst = self.get('instance', record['instance_token'])
            record['category_name'] = self.get('category', inst['category_token'])['name']

        # Decorate (adds short-cut) sample_data with sensor information.
        for record in self.sample_data:
            cs_record = self.get('calibrated_sensor', record['calibrated_sensor_token'])
            sensor_record = self.get('sensor', cs_record['sensor_token'])
            record['sensor_modality'] = sensor_record['modality']
            record['channel'] = sensor_record['channel']

        # Reverse-index samples with sample_data and annotations.
        for record in self.sample:
            record['data'] = {}
            record['anns'] = []

        for record in self.sample_data:
            if record['is_key_frame']:
                sample_record = self.get('sample', record['sample_token'])
                sample_record['data'][record['channel']] = record['token']

        for ann_record in self.sample_annotation:
            sample_record = self.get('sample', ann_record['sample_token'])
            sample_record['anns'].append(ann_record['token'])

        # Add reverse indices from log records to map records.
        if 'log_tokens' not in self.map[0].keys():
            raise Exception('Error: log_tokens not in map table. This code is not compatible with the teaser dataset.')
        log_to_map = dict()
        for map_record in self.map:
            for log_token in map_record['log_tokens']:
                log_to_map[log_token] = map_record['token']
        for log_record in self.log:
            log_record['map_token'] = log_to_map[log_record['token']]

        if verbose:
            print("Done reverse indexing in {:.1f} seconds.\n======".format(time.time() - start_time))

    def get(self, table_name: str, token: str) -> dict:
        """
        Returns a record from table in constant runtime.
        :param table_name: Table name.
        :param token: Token of the record.
        :return: Table record. See README.md for record details for each table.
        """
        assert table_name in self.table_names, "Table {} not found".format(table_name)

        return getattr(self, table_name)[self.getind(table_name, token)]

    def getind(self, table_name: str, token: str) -> int:
        """
        This returns the index of the record in a table in constant runtime.
        :param table_name: Table name.
        :param token: Token of the record.
        :return: The index of the record in table, table is an array.
        """
        return self._token2ind[table_name][token]

    def field2token(self, table_name: str, field: str, query) -> List[str]:
        """
        This function queries all records for a certain field value, and returns the tokens for the matching records.
        Warning: this runs in linear time.
        :param table_name: Table name.
        :param field: Field name. See README.md for details.
        :param query: Query to match against. Needs to type match the content of the query field.
        :return: List of tokens for the matching records.
        """
        matches = []
        for member in getattr(self, table_name):
            if member[field] == query:
                matches.append(member['token'])
        return matches

    def get_sample_data_path(self, sample_data_token: str) -> str:
        """ Returns the path to a sample_data. """

        sd_record = self.get('sample_data', sample_data_token)
        return osp.join(self.dataroot, sd_record['filename'])

    def get_sample_data(self, sample_data_token: str,
                        box_vis_level: BoxVisibility = BoxVisibility.ANY,
                        selected_anntokens: List[str] = None,
                        use_flat_vehicle_coordinates: bool = False) -> \
            Tuple[str, List[Box], np.array]:
        """
        Returns the data path as well as all annotations related to that sample_data.
        Note that the boxes are transformed into the current sensor's coordinate frame.
        :param sample_data_token: Sample_data token.
        :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
        :param selected_anntokens: If provided only return the selected annotation.
        :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
                                             aligned to z-plane in the world.
        :return: (data_path, boxes, camera_intrinsic <np.array: 3, 3>)
        """

        # Retrieve sensor & pose records
        sd_record = self.get('sample_data', sample_data_token)
        cs_record = self.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        sensor_record = self.get('sensor', cs_record['sensor_token'])
        pose_record = self.get('ego_pose', sd_record['ego_pose_token'])

        data_path = self.get_sample_data_path(sample_data_token)

        if sensor_record['modality'] == 'camera':
            cam_intrinsic = np.array(cs_record['camera_intrinsic'])
            imsize = (sd_record['width'], sd_record['height'])
        else:
            cam_intrinsic = None
            imsize = None

        # Retrieve all sample annotations and map to sensor coordinate system.
        if selected_anntokens is not None:
            boxes = list(map(self.get_box, selected_anntokens))
        else:
            boxes = self.get_boxes(sample_data_token)

        # Make list of Box objects including coord system transforms.
        box_list = []
        for box in boxes:
            if use_flat_vehicle_coordinates:
                # Move box to ego vehicle coord system parallel to world z plane.
                yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
                box.translate(-np.array(pose_record['translation']))
                box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
            else:
                # Move box to ego vehicle coord system.
                box.translate(-np.array(pose_record['translation']))
                box.rotate(Quaternion(pose_record['rotation']).inverse)

                #  Move box to sensor coord system.
                box.translate(-np.array(cs_record['translation']))
                box.rotate(Quaternion(cs_record['rotation']).inverse)

            if sensor_record['modality'] == 'camera' and not \
                    box_in_image(box, cam_intrinsic, imsize, vis_level=box_vis_level):
                continue

            box_list.append(box)

        return data_path, box_list, cam_intrinsic

    def get_box(self, sample_annotation_token: str) -> Box:
        """
        Instantiates a Box class from a sample annotation record.
        :param sample_annotation_token: Unique sample_annotation identifier.
        """
        record = self.get('sample_annotation', sample_annotation_token)
        return Box(record['translation'], record['size'], Quaternion(record['rotation']),
                   name=record['category_name'], token=record['token'])

    def get_boxes(self, sample_data_token: str) -> List[Box]:
        """
        Instantiates Boxes for all annotation for a particular sample_data record. If the sample_data is a
        keyframe, this returns the annotations for that sample. But if the sample_data is an intermediate
        sample_data, a linear interpolation is applied to estimate the location of the boxes at the time the
        sample_data was captured.
        :param sample_data_token: Unique sample_data identifier.
        """

        # Retrieve sensor & pose records
        sd_record = self.get('sample_data', sample_data_token)
        curr_sample_record = self.get('sample', sd_record['sample_token'])

        if curr_sample_record['prev'] == "" or sd_record['is_key_frame']:
            # If no previous annotations available, or if sample_data is keyframe just return the current ones.
            boxes = list(map(self.get_box, curr_sample_record['anns']))

        else:
            prev_sample_record = self.get('sample', curr_sample_record['prev'])

            curr_ann_recs = [self.get('sample_annotation', token) for token in curr_sample_record['anns']]
            prev_ann_recs = [self.get('sample_annotation', token) for token in prev_sample_record['anns']]

            # Maps instance tokens to prev_ann records
            prev_inst_map = {entry['instance_token']: entry for entry in prev_ann_recs}

            t0 = prev_sample_record['timestamp']
            t1 = curr_sample_record['timestamp']
            t = sd_record['timestamp']

            # There are rare situations where the timestamps in the DB are off so ensure that t0 < t < t1.
            t = max(t0, min(t1, t))

            boxes = []
            for curr_ann_rec in curr_ann_recs:

                if curr_ann_rec['instance_token'] in prev_inst_map:
                    # If the annotated instance existed in the previous frame, interpolate center & orientation.
                    prev_ann_rec = prev_inst_map[curr_ann_rec['instance_token']]

                    # Interpolate center.
                    center = [np.interp(t, [t0, t1], [c0, c1]) for c0, c1 in zip(prev_ann_rec['translation'],
                                                                                 curr_ann_rec['translation'])]

                    # Interpolate orientation.
                    rotation = Quaternion.slerp(q0=Quaternion(prev_ann_rec['rotation']),
                                                q1=Quaternion(curr_ann_rec['rotation']),
                                                amount=(t - t0) / (t1 - t0))

                    box = Box(center, curr_ann_rec['size'], rotation, name=curr_ann_rec['category_name'],
                              token=curr_ann_rec['token'])
                else:
                    # If not, simply grab the current annotation.
                    box = self.get_box(curr_ann_rec['token'])

                boxes.append(box)
        return boxes

    def box_velocity(self, sample_annotation_token: str, max_time_diff: float = 1.5) -> np.ndarray:
        """
        Estimate the velocity for an annotation.
        If possible, we compute the centered difference between the previous and next frame.
        Otherwise we use the difference between the current and previous/next frame.
        If the velocity cannot be estimated, values are set to np.nan.
        :param sample_annotation_token: Unique sample_annotation identifier.
        :param max_time_diff: Max allowed time diff between consecutive samples that are used to estimate velocities.
        :return: <np.float: 3>. Velocity in x/y/z direction in m/s.
        """

        current = self.get('sample_annotation', sample_annotation_token)
        has_prev = current['prev'] != ''
        has_next = current['next'] != ''

        # Cannot estimate velocity for a single annotation.
        if not has_prev and not has_next:
            return np.array([np.nan, np.nan, np.nan])

        if has_prev:
            first = self.get('sample_annotation', current['prev'])
        else:
            first = current

        if has_next:
            last = self.get('sample_annotation', current['next'])
        else:
            last = current

        pos_last = np.array(last['translation'])
        pos_first = np.array(first['translation'])
        pos_diff = pos_last - pos_first

        time_last = 1e-6 * self.get('sample', last['sample_token'])['timestamp']
        time_first = 1e-6 * self.get('sample', first['sample_token'])['timestamp']
        time_diff = time_last - time_first

        if has_next and has_prev:
            # If doing centered difference, allow for up to double the max_time_diff.
            max_time_diff *= 2

        if time_diff > max_time_diff:
            # If time_diff is too big, don't return an estimate.
            return np.array([np.nan, np.nan, np.nan])
        else:
            return pos_diff / time_diff
        
        return


class PCDetSceneData(NuScenesDataset):
    """Class for PCDet data preprocessing of a single scene"""

    def __init__(self, sample_tokens, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        # root_path = (root_path if root_path is not None else Path(dataset_cfg.DATA_PATH)) / dataset_cfg.VERSION
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        # self.infos = []
        # self.include_nuscenes_data(self.mode)
        # if self.training and self.dataset_cfg.get('BALANCED_RESAMPLING', False):
        #     self.infos = self.balanced_infos_resampling(self.infos)
        scene_infos = []
        # print(self.infos)
        for info in self.infos:
            #print(info['token'])
            if info['token'] in sample_tokens:
                scene_infos.append(info)
        
        self.infos = scene_infos