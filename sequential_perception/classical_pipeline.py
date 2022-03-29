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
        self.predictor = PipelineCoverNetModule(nuscenes,
                                                eps_sets_path=predict_config['EPS_SETS'],
                                                weights_path=predict_config['MODEL_CKPT'])
        return

    def __call__(self, data_dicts: List[Dict], pred_tokens: List[str], reset=True):
        detections = self.run_detection(data_dicts)
        tracks = self.run_tracking(detections, reset)
        predictions = self.run_prediction(tracks, pred_tokens)
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
        predictions = self.predictor(tokens, tracks)

        if predictions == None:
            prediction_dicts = []
        else:
            prediction_dicts = [p.serialize() for p in predictions]

        return prediction_dicts

    def reset(self):
        self.tracker.reset()



def build_pipeline(pipeline_config: Dict,
                   nuscenes: NuScenes) -> ClassicalPerceptionPipeline:

    model_config_path = pipeline_config['DETECTION']['MODEL_CONFIG']
    pcdet_config = cfg_from_yaml_file(model_config_path, cfg)
    if 'train' in pipeline_config['NUSCENES_SPLIT']:
        pcdet_config['DATA_CONFIG']['INFO_PATH']['test'] = pcdet_config['DATA_CONFIG']['INFO_PATH']['train']

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
        self.detector = detector
        self.tracker = tracker
        self.predictor = predictor
        return

    def __call__(self, data_dicts: List[Dict], pred_tokens: List[str], reset=True):
        detections = self.run_detection(data_dicts)
        tracks = self.run_tracking(detections, batch_mode=reset)
        predictions = self.run_prediction(tracks, pred_tokens)
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
        if len(tokens) == 0 or tracks['results'] == {}:
            return []

        predictions = self.predictor(tokens, tracks)

        if predictions == None:
            prediction_dicts = []
        else:
            prediction_dicts = [p.serialize() for p in predictions]

        return prediction_dicts

    def reset(self):
        self.tracker.reset()

