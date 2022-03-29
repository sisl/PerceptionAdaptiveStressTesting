import pickle
from torch.multiprocessing import Pool
import argparse
import time
import random
import yaml
from pathlib import Path
import torch
import numpy as np

from nuscenes import NuScenes
from nuscenes.prediction.models.backbone import ResNetBackbone
from nuscenes.prediction.models.covernet import CoverNet

from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset # as PCDetNuScenesDataset
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from pcdet.models import build_network
from sequential_perception.classical_pipeline import PerceptionPipeline

from sequential_perception.datasets import NuScenesScene, PCDetSceneData
from sequential_perception.detectors import PCDetModule
from sequential_perception.iso_simulator import ISOPerceptionSimulator
from sequential_perception.kalman_tracker import MultiObjectTrackByDetection
from sequential_perception.nuscenes_utils import get_ordered_samples
from sequential_perception.predictors import CoverNetPredictModule


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_detector(pipeline_config, nusc):
    model_config_path = pipeline_config['DETECTION']['MODEL_CONFIG']
    pcdet_config = cfg_from_yaml_file(model_config_path, cfg)
    if 'train' in pipeline_config['NUSCENES_SPLIT']:
        pcdet_config['DATA_CONFIG']['INFO_PATH']['test'] = pcdet_config['DATA_CONFIG']['INFO_PATH']['train']

    detection_config = pipeline_config['DETECTION']
    #scene_token = nusc.scene[1]['token']
    
    scene_sample_tokens = get_ordered_samples(nusc, nusc.scene[0]['token'])
    logger = common_utils.create_logger()
    pcdet_scene = PCDetSceneData(scene_sample_tokens,dataset_cfg=pcdet_config.DATA_CONFIG, class_names=pcdet_config.CLASS_NAMES, training=False,
        root_path=Path(pcdet_config.DATA_CONFIG.DATA_PATH), logger=logger)
    
    model = build_network(model_cfg=pcdet_config.MODEL, num_class=len(pcdet_config.CLASS_NAMES), dataset=pcdet_scene)
    model.load_params_from_file(filename=detection_config['MODEL_CKPT'], logger=logger, to_cpu=True)

    # build module
    detector = PCDetModule(model, nusc)

    return detector

def build_tracker(pipeline_config, nusc):
    tracking_config = pipeline_config['TRACKING']
    tracker = MultiObjectTrackByDetection(nusc)
    return tracker

def build_predictor(pipeline_config, nusc):
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

    predictor = CoverNetPredictModule(covernet, trajectories, nusc)

    return predictor

def build_pipeline(pipeline_config, detector, predictor):
    tracker = build_tracker(pipeline_config, detector.nuscenes)
    pipeline = PerceptionPipeline(detector, tracker, predictor)

    return pipeline

def run(args):
    pipeline_config = args[0]
    scene_token = args[1]
    pointpillars = args[2]
    covernet = args[3]
    trajectories = args[4]

    dataroot = '/mnt/hdd/data/sets/nuscenes/v1.0-trainval'
    version = 'v1.0-trainval'
    nusc_scene_root = '/mnt/hdd/data/sets/nuscenes/v1.0-trainval/exported_scenes'
    nusc_scene_path = Path(nusc_scene_root)
    with open(nusc_scene_path / (scene_token+'.pkl'), 'rb') as f:
        nusc_scene = pickle.load(f)



    infos_scene_root = '/mnt/hdd/data/sets/nuscenes/v1.0-trainval/exported_infos'
    infos_scene_path = Path(infos_scene_root)
    with open(infos_scene_path / (scene_token+'.pkl'), 'rb') as f:
        scene_infos_list = pickle.load(f)

    # Detector
    detector = PCDetModule(pointpillars, nusc_scene)

    # Tracker  
    tracking_config = pipeline_config['TRACKING']
    tracker = MultiObjectTrackByDetection(nusc_scene)

    # Predictor
    predictor = CoverNetPredictModule(covernet, trajectories, nusc_scene)

    # Pipeline
    pipeline = PerceptionPipeline(detector, tracker, predictor)



    #fog_density = 0.005
    # fog_density = [0.0025, 0.01, 0.02]
    sim_args = {'eval_metric': 'MinFDEK',
                'fog_density': 0.005,
                'scatter_fraction': 0.05,
                'eval_k': 5,
                'eval_metric_th': 20.0}

    try:
        #sim = ScenePerceptionSimulator(nusc_scene, pipeline, scene_infos_list, scene_token=scene_token, eval_metric='MinFDEK', fog_density=fog_density, scatter_fraction=0.05, eval_k=5, eval_metric_th=15.0, no_pred_failure=False)
        sim = ISOPerceptionSimulator(nusc_scene,
                                        pipeline,
                                        scene_infos_list,
                                        scene_token=scene_token,
                                        eval_metric=sim_args['eval_metric'],
                                        fog_density=sim_args['fog_density'],
                                        scatter_fraction=sim_args['scatter_fraction'],
                                        eval_k=sim_args['eval_k'],
                                        eval_metric_th=sim_args['eval_metric_th'])
    except Exception as e:
        print(e)
        print('Failed to initialize fog simulator for scene {}. Probably because there is nothing to track/predict.'.format(scene_token))
        return {}

    print('Starting ISO on scene {}'.format(scene_token))
    start = time.time()
    sim.simulate()
    print('Finished ISO on scene {}. took {} sec'.format(scene_token, time.time() - start))


    base_log_dir = Path('/mnt/hdd/hdelecki/ford_ws/output/baseline_baseline/' + scene_token)
    base_log_dir.mkdir(parents=True, exist_ok=True)

    if sim.is_goal():
        #save off results now!
        summary_path = base_log_dir / 'summary.pkl'
        with open(summary_path, 'wb') as f:
            #json.dump(summary, f)
            pickle.dump(sim.sim_log, f)

        data_path = base_log_dir / 'data.pkl'
        with open(data_path, 'wb') as f:
            #json.dump(summary, f)
            pickle.dump(sim.failure_perception_data, f)

        return sim.sim_log
    else:
        return {}


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--config', type=str, default='/mnt/hdd/hdelecki/ford_ws/src/SequentialPerceptionPipeline/configs/detection_config.yaml',
                        help='specify the config for detection')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        detection_config = yaml.load(f)

    return args, detection_config



def load_pcdet_config(detection_config):
    model_config_path = detection_config['MODEL_CONFIG']
    pcdet_config = cfg_from_yaml_file(model_config_path, cfg)
    if 'train' in detection_config['NUSCENES_SPLIT']:
        pcdet_config['DATA_CONFIG']['INFO_PATH']['test'] = pcdet_config['DATA_CONFIG']['INFO_PATH']['train']

    return pcdet_config

if __name__ == '__main__':

    torch.multiprocessing.set_start_method('spawn')
    pipeline_config_path = '/mnt/hdd/hdelecki/ford_ws/src/SequentialPerceptionPipeline/configs/pipeline_config.yaml'
    with open(pipeline_config_path, 'r') as f:
        pipeline_config = yaml.load(f, yaml.SafeLoader)



    # Create PointPillars 
    logger = common_utils.create_logger()
    args, detection_config = parse_config()
    cfg = load_pcdet_config(detection_config)
    dataset = NuScenesDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(cfg.DATA_CONFIG.DATA_PATH), logger=logger
    )
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=detection_config['MODEL_CKPT'], logger=logger, to_cpu=True)


    # Create CoverNet + Trajectories
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


    all_scene_tokens = []
    scene_fname = '/mnt/hdd/hdelecki/ford_ws/src/SequentialPerceptionPipeline/configs/val_nonweather_scene_tokens.txt'
    with open(scene_fname, 'r') as f:
        for line in f:
            token = line.rstrip()
            all_scene_tokens.append(token)
    
    start = time.time()
    scene_idxs = range(len(all_scene_tokens))
    pipelines = [[pipeline_config, all_scene_tokens[i], model, covernet, trajectories] for i in scene_idxs]

    pool = Pool(3)
    sim_logs = pool.map(run, pipelines) 


    print('Took {} sec'.format(time.time() - start))