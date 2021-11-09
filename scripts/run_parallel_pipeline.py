import argparse
import time
import pickle

import yaml
import json
from pathlib import Path

import torch

from nuscenes import NuScenes
from nuscenes.prediction.models.backbone import ResNetBackbone
from nuscenes.prediction.models.covernet import CoverNet
from nuscenes.eval.prediction.splits import get_prediction_challenge_split

from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset # as PCDetNuScenesDataset
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from pcdet.models import build_network, detectors
from sequential_perception.classical_pipeline import PerceptionPipeline

from sequential_perception.datasets import NuScenesScene, PCDetSceneData
from sequential_perception.detectors import PCDetModule
from sequential_perception.kalman_tracker import MultiObjectTrackByDetection
from sequential_perception.nuscenes_utils import get_ordered_samples
from sequential_perception.predictors import CoverNetPredictModule


def main(pipeline_config):
    
    # Load PCDet config
    model_config_path = pipeline_config['DETECTION']['MODEL_CONFIG']
    pcdet_config = cfg_from_yaml_file(model_config_path, cfg)
    if 'train' in pipeline_config['NUSCENES_SPLIT']:
        pcdet_config['DATA_CONFIG']['INFO_PATH']['test'] = pcdet_config['DATA_CONFIG']['INFO_PATH']['train']
    #config = cfg_from_yaml_file(model_config_path, cfg)
    
    # Create PCDet info dataset
    # data_path = pipeline_config['NUSCENES_DATAROOT']
    # pcdet_data_path = data_path.strip(pipeline_config['NUSCENES_VERSION'])
    # logger = common_utils.create_logger()
    # pcdet_infos = NuScenesDataset(dataset_cfg=pcdet_config.DATA_CONFIG, class_names=pcdet_config.CLASS_NAMES, training=False,
    #                                root_path=Path(pcdet_data_path), logger=logger)

    # Create NuScenes
    dataroot = '/mnt/hdd/data/sets/nuscenes/v1.0-mini'
    version = 'v1.0-mini'
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)

    ######## Create Detector ########
    detection_config = pipeline_config['DETECTION']
    scene_token = nusc.scene[1]['token']
    nusc_scene = NuScenesScene(dataroot, nusc, scene_token)


    scene_sample_tokens = get_ordered_samples(nusc, scene_token)
    logger = common_utils.create_logger()
    pcdet_scene = PCDetSceneData(scene_sample_tokens,dataset_cfg=pcdet_config.DATA_CONFIG, class_names=pcdet_config.CLASS_NAMES, training=False,
        root_path=Path(pcdet_config.DATA_CONFIG.DATA_PATH), logger=logger)
    
    model = build_network(model_cfg=pcdet_config.MODEL, num_class=len(pcdet_config.CLASS_NAMES), dataset=pcdet_scene)
    model.load_params_from_file(filename=detection_config['MODEL_CKPT'], logger=logger, to_cpu=True)

    # build module
    detector = PCDetModule(model, nusc_scene)


    ######## Create Tracker ########
    tracking_config = pipeline_config['TRACKING']
    tracker = MultiObjectTrackByDetection(nusc_scene)

    ######## Create Predictor ########
    predict_config = pipeline_config['PREDICTION']
    # CoverNet
    weights_path=predict_config['MODEL_CKPT']
    backbone = ResNetBackbone('resnet50')
    covernet = CoverNet(backbone, num_modes=64)
    covernet.load_state_dict(torch.load(weights_path))

    # Trajectories
    eps_sets_path=predict_config['EPS_SETS']
    trajectories = pickle.load(open(eps_sets_path, 'rb'))
    trajectories = torch.Tensor(trajectories)

    predictor = CoverNetPredictModule(covernet, trajectories, nusc_scene)


    ######## Create Pipeline ########
    # pipeline = ClassicalPerceptionPipeline(pipeline_config, nuscenes, pcdet_infos)
    pipeline = PerceptionPipeline(detector, tracker, predictor)

    ## Detection
    all_data_dicts = [pcdet_scene[i] for i in range(len(pcdet_scene))]
    start = time.time()
    print('Running Detection ...')
    detections = pipeline.run_detection(all_data_dicts)
    print('Done. Detection took {} sec'.format(time.time() - start))

    detection_results_path = Path(pipeline_config['DETECTION']['RESULTS'])
    detection_results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pipeline_config['DETECTION']['RESULTS'], 'w') as f:
        json.dump(detections, f)


    ## Tracking
    start = time.time()
    print('Running Tracking ...')
    tracks = pipeline.run_tracking(detections, batch_mode=True)
    print('Done. Tracking took {} sec'.format(time.time() - start))


    # Tracking evaluation
    tracking_results_path = Path(pipeline_config['TRACKING']['RESULTS'])
    tracking_results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pipeline_config['TRACKING']['RESULTS'], 'w') as f:
        json.dump(tracks, f)
        
    # tracking_eval_cfg = config_factory('tracking_nips_2019')
    # eval_set = 'mini_val'

    # output_dir='/scratch/hdelecki/ford/output/tracking/kalman/nusc_tracking_evaluation/kalman_tracker_mini'
    # nusc_tracking_eval = TrackingEval(config=tracking_eval_cfg, result_path=pipeline_config['TRACKING']['RESULTS'], 
    #                                   eval_set=eval_set, nusc_version=pipeline_config['NUSCENES_VERSION'],
    #                                   output_dir=output_dir, nusc_dataroot=pipeline_config['NUSCENES_DATAROOT'],
    #                                   render_classes=None, verbose=True)
    # nusc_tracking_eval.main(render_curves=0)    

    ## Prediction
    print('Running Predicton ...')
    predict_tokens = get_prediction_challenge_split(pipeline_config['NUSCENES_SPLIT'], dataroot=pipeline_config['NUSCENES_DATAROOT'])
    predict_tokens = [t for t in predict_tokens if t.split('_')[1] in scene_sample_tokens]

    start = time.time()
    preds = pipeline.run_prediction(tracks, predict_tokens)
    print('Done. Prediction took {} sec'.format(time.time() - start))
    print('Done')
    print(len(preds))

    prediction_results_path = Path(pipeline_config['PREDICTION']['RESULTS'])
    prediction_results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pipeline_config['PREDICTION']['RESULTS'], 'w') as f:
        json.dump(preds, f)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--config', type=str, default='./configs/pipeline_config.yaml',
                        help='specify the pipeline config')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        pipeline_config = yaml.load(f)
    
    main(pipeline_config)
