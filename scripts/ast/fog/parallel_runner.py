import pickle
#from multiprocessing import Pool
from torch.multiprocessing import Pool
from pcdet.models.detectors import pointpillar
#from joblib import Parallel, delayed
import yaml
from runner_mcts_fog import runner as mcts_runner

import argparse
import time
import pickle
import random

import yaml
import json
from pathlib import Path

import torch
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
from sequential_perception.scene_simulator import FogScenePerceptionSimulator
# from sequential_perception.ast import PerceptionSimWrapper


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

def build_tracker(peipeline_config, nusc):
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

#def run(pipeline_config, scene_token, pointpillars, covernet, trajectories):
def run(args):
    pipeline_config = args[0]
    scene_token = args[1]
    pointpillars = args[2]
    covernet = args[3]
    trajectories = args[4]
    # Create simulator
    # Create pipeline
    #pipeline = build_pipeline(pipeline_config, detector, predictor)
    dataroot = '/mnt/hdd/data/sets/nuscenes/v1.0-mini'
    version = 'v1.0-mini'
    #scene_token = 'fcbccedd61424f1b85dcbf8f897f9754'
    nusc_scene_root = '/mnt/hdd/data/sets/nuscenes/v1.0-trainval/exported_scenes'
    nusc_scene_path = Path(nusc_scene_root)
    with open(nusc_scene_path / (scene_token+'.pkl'), 'rb') as f:
        nusc_scene = pickle.load(f)

    # ordered_samples = get_ordered_samples(nusc_scene, scene_token)
    # pcdet_infos = []
    # #for data_dict in pointpillars.dataset:
    # for i in range(len(pointpillars.dataset)):
    #     data_dict = pointpillars.dataset[i]
    #     if data_dict['metadata']['token'] in ordered_samples:
    #         pcdet_infos.append(data_dict)
    # pointpillars.dataset.infos = pcdet_infos

    



    # Detector
    detector = PCDetModule(pointpillars, nusc_scene)

    # Tracker  
    tracking_config = pipeline_config['TRACKING']
    tracker = MultiObjectTrackByDetection(nusc_scene)

    # Predictor
    predictor = CoverNetPredictModule(covernet, trajectories, nusc_scene)

    # Pipeline
    pipeline = PerceptionPipeline(detector, tracker, predictor)




    sim = FogScenePerceptionSimulator(nusc_scene, pipeline, eval_metric='MinFDEK', fog_density=0.005, scatter_fraction=0.05, eval_k=5, eval_metric_th=20.0)
    
    
    
    # Overall settings
    max_path_length = sim.horizon
    s_0 = [0]
    base_log_dir = './' + scene_token
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
    runner_args = {'n_epochs': 100,
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
                        'n_itr': 50,
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

    mcts_runner(
        sim,
        mcts_type=mcts_type,
        env_args=env_args,
        run_experiment_args=run_experiment_args,
        sim_args=sim_args,
        #perception_args=pipeline_args,
        reward_args=reward_args,
        spaces_args=spaces_args,
        algo_args=mcts_algo_args,
        runner_args=runner_args,
        bpq_args=mcts_bpq_args,
        sampler_args=mcts_sampler_args,
        save_expert_trajectory=True,
    )




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



    # scene_token = 'fcbccedd61424f1b85dcbf8f897f9754'
    # run(pipeline_config, scene_token, model, covernet, trajectories)






    # scene_token = 'fcbccedd61424f1b85dcbf8f897f9754'
    # target_instance = '045cd82a77a1472499e8c15100cb5ff3'
    # scene_token ='c5224b9b454b4ded9b5d2d2634bbda8a'
    # target_instance = '88e4abb15ade4097a1689225c13399ec'
    # metric = 'MinFDEK'
    # k = 1
    # threshold = 20
    # drop_likelihood = 0.1
    # max_range = 35
    # pipeline_args = {'pipeline_config': pipeline_config,
    #                  'scene_token': scene_token,
    #                  'target_instance': target_instance,
    #                  'drop_likelihood': drop_likelihood,
    #                  'eval_metric': metric,
    #                  'eval_k': k,
    #                  'eval_metric_th': threshold,
    #                  'no_pred_failure': False,
    #                  'max_range': max_range
    #                 }

    # pipelines = 4*[pipeline_args]

    #scene_tokens = 2*[[pipeline_config, 'fcbccedd61424f1b85dcbf8f897f9754', model, covernet, trajectories]]
    #run(pipeline_config, scene_token, model, covernet, trajectories)
    start = time.time()
    pipelines = 1*[[pipeline_config, 'fcbccedd61424f1b85dcbf8f897f9754', model, covernet, trajectories]]
    # print(pipelines)


    #dummy = Parallel(n_jobs=2, require='sharedmem')(delayed(run)(pipeline_config, token, model, covernet, trajectories) for token in scene_tokens)

    # pool = Pool(1)
    # pool.map(run, pipelines) 
    run(pipelines[0])

    print('Took {} sec'.format(time.time() - start))