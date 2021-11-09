from pathlib import Path
import time
from typing import Dict, List
import numpy as np
from nuscenes import NuScenes

from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset
from pcdet.config import cfg_from_yaml_file, cfg
from pcdet.utils import common_utils

from sequential_perception.detectors import OpenPCDetector, PCDetModule
from sequential_perception.kalman_tracker import MultiObjectTrackByDetection
from sequential_perception.predictors import CoverNetPredictModule, CoverNetTrackPredictor, PipelineCoverNetModule
from sequential_perception.predict_helper_tracks import TrackingResultsPredictHelper


class ClassicalPerceptionPipeline:
    
    def __init__(self, config: Dict, nuscenes: NuScenes, pcdet_dataset: NuScenesDataset):
        self.config = config
        self.nuscenes = nuscenes
        self.pcdet_dataset = pcdet_dataset

        detection_config = config['DETECTION']
        self.detector = OpenPCDetector(detection_config['MODEL_CONFIG'],
                                       detection_config['MODEL_CKPT'],
                                       nuscenes,
                                       pcdet_dataset)
        
        tracking_config = config['TRACKING']
        self.tracker = MultiObjectTrackByDetection(nuscenes)
        
        predict_config = config['PREDICTION']
        #predict_helper = TrackingResultsPredictHelper(self.nuscenes, {})
        # self.predictor = CoverNetTrackPredictor(predict_helper,
        #                                         path_to_traj_sets=predict_config['EPS_SETS'],
        #                                         path_to_weights=predict_config['MODEL_CKPT'],
        #                                         use_cuda=True)
        self.predictor = PipelineCoverNetModule(nuscenes,
                                                eps_sets_path=predict_config['EPS_SETS'],
                                                weights_path=predict_config['MODEL_CKPT'])
        return

    def __call__(self, data_dicts: List[Dict], pred_tokens: List[str], reset=True):
        detections = self.run_detection(data_dicts)
        tracks = self.run_tracking(detections, reset)
        predictions = self.run_prediction(tracks, pred_tokens)
        # return detections, tracks, predictions
        return detections, tracks, predictions

    def run_detection(self, data_dicts: List[Dict]):
        return self.detector(data_dicts)

    def run_tracking(self, detections: Dict, batch_mode=True):
        if batch_mode:
            return self.tracker(detections, reset=True)
        else:
            return self.tracker.update(detections)

    def run_prediction(self, tracks: Dict, tokens: List[str]):
        if len(tokens) == 0:
            return []

        # setup_start = time.time()
        # predict_helper = TrackingResultsPredictHelper(self.nuscenes, tracks['results'])

        # print('Setup time: {}'.format(time.time() - setup_start))
        

        # p_start = time.time()

        predictions = self.predictor(tokens, tracks)
        # print('Pred loop time: {}'.format(time.time() - p_start))

        if predictions == None:
            prediction_dicts = []
        else:
            prediction_dicts = [p.serialize() for p in predictions]
        
        return prediction_dicts

    # def run_prediction(self, tracks: Dict, tokens: List[str]):
    #     if len(tokens) == 0:
    #         return []

    #     setup_start = time.time()
    #     predict_helper = TrackingResultsPredictHelper(self.nuscenes, tracks['results'])
    #     predict_config = self.config['PREDICTION']
    #     predictor = CoverNetTrackPredictor(predict_helper,
    #                                        path_to_traj_sets=predict_config['EPS_SETS'],
    #                                        path_to_weights=predict_config['MODEL_CKPT'],
    #                                        use_cuda=True)
    #     print('Setup time: {}'.format(time.time() - setup_start))
        

    #     p_start = time.time()

    #     predictions = []
    #     for token in tokens:
    #         pred = predictor(token)
    #         if pred != None:
    #             predictions.append(pred)
    #     print('Pred loop time: {}'.format(time.time() - p_start))

    #     prediction_dicts = [p.serialize() for p in predictions]
        
    #     return prediction_dicts

    def reset(self):
        self.tracker.reset()



def build_pipeline(pipeline_config: Dict,
                   nuscenes: NuScenes) -> ClassicalPerceptionPipeline:

    model_config_path = pipeline_config['DETECTION']['MODEL_CONFIG']
    pcdet_config = cfg_from_yaml_file(model_config_path, cfg)
    if 'train' in pipeline_config['NUSCENES_SPLIT']:
        pcdet_config['DATA_CONFIG']['INFO_PATH']['test'] = pcdet_config['DATA_CONFIG']['INFO_PATH']['train']
        #pcdet_config['DATA_CONFIG']['INFO_PATH']['val'] = pcdet_config['DATA_CONFIG']['INFO_PATH']['train']
    #config = cfg_from_yaml_file(model_config_path, cfg)
    
    # Create PCDet info dataset
    data_path = pipeline_config['NUSCENES_DATAROOT']
    pcdet_data_path = data_path.strip(pipeline_config['NUSCENES_VERSION'])
    logger = common_utils.create_logger()
    pcdet_infos = NuScenesDataset(dataset_cfg=pcdet_config.DATA_CONFIG, class_names=pcdet_config.CLASS_NAMES, training=False,
                                   root_path=Path(pcdet_data_path), logger=logger)

    # Create Pipeline
    pipeline = ClassicalPerceptionPipeline(pipeline_config, nuscenes, pcdet_infos)

    return pipeline


class PerceptionPipeline:
    
    def __init__(self, detector:PCDetModule, tracker:MultiObjectTrackByDetection, predictor:CoverNetPredictModule):
        # self.config = config
        # self.nuscenes = nuscenes
        self.detector = detector
        self.tracker = tracker
        self.predictor = predictor
        return

    def __call__(self, data_dicts: List[Dict], pred_tokens: List[str], reset=True):
        detections = self.run_detection(data_dicts)
        tracks = self.run_tracking(detections, batch_mode=reset)
        predictions = self.run_prediction(tracks, pred_tokens)
        # return detections, tracks, predictions
        if reset:
            self.reset()
        return detections, tracks, predictions

    def run_detection(self, data_dicts: List[Dict]):
        return self.detector(data_dicts)

    def run_tracking(self, detections: Dict, batch_mode=True):
        if batch_mode:
            return self.tracker(detections, reset=True)
        else:
            return self.tracker.update(detections)

    def run_prediction(self, tracks: Dict, tokens: List[str]):
        if len(tokens) == 0:
            return []

        # setup_start = time.time()
        # predict_helper = TrackingResultsPredictHelper(self.nuscenes, tracks['results'])

        # print('Setup time: {}'.format(time.time() - setup_start))
        

        # p_start = time.time()

        predictions = self.predictor(tokens, tracks)
        # print('Pred loop time: {}'.format(time.time() - p_start))

        if predictions == None:
            prediction_dicts = []
        else:
            prediction_dicts = [p.serialize() for p in predictions]
        
        return prediction_dicts

    def reset(self):
        self.tracker.reset()
    
