from .runner_mcts import runner as mcts_runner

import pickle
import yaml
import argparse
import time
import random
from pathlib import Path
import torch
from torch.multiprocessing import Pool
import numpy as np

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
from sequential_perception.scene_simulator import ScenePerceptionSimulator

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

    try:
        sim = ScenePerceptionSimulator(nusc_scene, pipeline, scene_infos_list, scene_token=scene_token, eval_metric='MinFDEK', density=[5, 10, 20],  eval_k=5, eval_metric_th=15.0, no_pred_failure=False)
    except Exception as e:
        print(e)
        print('Failed to initialize fog simulator for scene {}. Probably because there is nothing to track/predict.'.format(scene_token))
        return 0
    
    
    # Overall settings
    max_path_length = sim.horizon
    s_0 = [0]
    base_log_dir = '/mnt/hdd/hdelecki/ford_ws/output/ast/' + scene_token
    # experiment settings
    run_experiment_args = {'snapshot_mode': 'last',
                           'snapshot_gap': 1,
                           'log_dir': None,
                           'exp_name': None,
                           'seed': 0,
                           'n_parallel': 1,
                           'tabular_log_file': 'progress.csv'
                           }

    # runner settings
    runner_args = {'n_epochs':  50,
                   'batch_size': 50,
                   'plot': False
                   }

    # env settings
    env_args = {'id': 'ast_toolbox:Perception',
                'blackbox_sim_state': True,
                'open_loop': False,
                'fixed_init_state': True,
                's_0': s_0,
                }

    sim_args = {'blackbox_sim_state': True,
                'open_loop': False,
                'fixed_initial_state': True,
                'max_path_length': max_path_length,
                }

    # reward settings
    reward_args = {'use_heuristic': False}

    # spaces settings
    spaces_args = {'rsg_length': 4}

    # MCTS Settings ----------------------------------------------------------------------
    mcts_type = 'mcts'

    mcts_sampler_args = {}

    mcts_algo_args = {'max_path_length': max_path_length,
                        'stress_test_mode': 2,
                        'ec': 100.0,
                        'n_itr': 100,
                        'k': 0.5,
                        'alpha': 0.5,
                        'clear_nodes': True,
                        'log_interval': 50,
                        'plot_tree': False,
                        'plot_path': None,
                        'log_dir': base_log_dir + '/mcts',
                        }

    mcts_bpq_args = {'N': 3}

    # MCTS settings
    run_experiment_args['log_dir'] = base_log_dir + '/mcts'
    run_experiment_args['exp_name'] = 'mcts'

    mcts_algo_args['max_path_length'] = max_path_length
    mcts_algo_args['log_dir'] = run_experiment_args['log_dir']
    mcts_algo_args['plot_path'] = run_experiment_args['log_dir']

    start = time.time()
    print('Starting AST on scene {}'.format(scene_token))
    mcts_runner(
        sim,
        mcts_type=mcts_type,
        env_args=env_args,
        run_experiment_args=run_experiment_args,
        sim_args=sim_args,
        reward_args=reward_args,
        spaces_args=spaces_args,
        algo_args=mcts_algo_args,
        runner_args=runner_args,
        bpq_args=mcts_bpq_args,
        sampler_args=mcts_sampler_args,
        save_expert_trajectory=True,
    )
    print('Finished AST for scene {}. Took {} sec'.format(scene_token, time.time() - start))

    return 1



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
    weights_path=predict_config['MODEL_CKPT']
    backbone = ResNetBackbone('resnet50')
    covernet = CoverNet(backbone, num_modes=64)
    covernet.load_state_dict(torch.load(weights_path))

    # Trajectories
    eps_sets_path=predict_config['EPS_SETS']
    trajectories = pickle.load(open(eps_sets_path, 'rb'))
    trajectories = torch.Tensor(trajectories)

    # Load scenes
    all_scene_tokens = []
    scene_fname = '/mnt/hdd/hdelecki/ford_ws/src/SequentialPerceptionPipeline/configs/val_noweather_scene_tokens.txt'
    with open(scene_fname, 'r') as f:
        for line in f:
            token = line.rstrip()
            all_scene_tokens.append(token)
    
    start = time.time()
    scene_idxs = range(len(all_scene_tokens))
    pipelines = [[pipeline_config, all_scene_tokens[i], model, covernet, trajectories] for i in scene_idxs]

    pool = Pool(3)
    pool.map(run, pipelines)

    print('Took {} sec'.format(time.time() - start))