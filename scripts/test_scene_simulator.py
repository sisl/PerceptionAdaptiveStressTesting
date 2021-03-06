import argparse
import time
import pickle
import random

import yaml
import json
from pathlib import Path

import torch
import numpy as np
import  matplotlib.pyplot as plt

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
from sequential_perception.ast import PerceptionSimWrapper



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
    


def main(pipeline_config):
    set_random_seed(0)


    # Load PCDet config
    # model_config_path = pipeline_config['DETECTION']['MODEL_CONFIG']
    # pcdet_config = cfg_from_yaml_file(model_config_path, cfg)
    # if 'train' in pipeline_config['NUSCENES_SPLIT']:
    #     pcdet_config['DATA_CONFIG']['INFO_PATH']['test'] = pcdet_config['DATA_CONFIG']['INFO_PATH']['train']
    #config = cfg_from_yaml_file(model_config_path, cfg)
    
    # Create PCDet info dataset
    # data_path = pipeline_config['NUSCENES_DATAROOT']
    # pcdet_data_path = data_path.strip(pipeline_config['NUSCENES_VERSION'])
    # logger = common_utils.create_logger()
    # pcdet_infos = NuScenesDataset(dataset_cfg=pcdet_config.DATA_CONFIG, class_names=pcdet_config.CLASS_NAMES, training=False,
    #                                root_path=Path(pcdet_data_path), logger=logger)

    # Create NuScenes

    dataroot = '/mnt/hdd/data/sets/nuscenes/v1.0-trainval'
    version = 'v1.0-trainval'
    scene_token = 'fcbccedd61424f1b85dcbf8f897f9754'
    nusc_scene_root = '/mnt/hdd/data/sets/nuscenes/v1.0-trainval/exported_scenes'
    nusc_scene_path = Path(nusc_scene_root)
    with open(nusc_scene_path / (scene_token+'.pkl'), 'rb') as f:
        nusc_scene = pickle.load(f)

    ## Create detector and Prediction modules
    detector = build_detector(pipeline_config, nusc_scene)
    predictor = build_predictor(pipeline_config, nusc_scene)

    # Create pipeline
    pipeline = build_pipeline(pipeline_config, detector, predictor)


    sim_args = {'eval_metric': 'MinFDEK',
                'fog_density': 0.005,
                'scatter_fraction': 0.05,
                'eval_k': 5,
                'eval_metric_th': 20.0}
    # sim = FogScenePerceptionSimulator(nusc_scene, pipeline, eval_metric='MinFDEK', fog_density=0.005, scatter_fraction=0.05, eval_k=1, eval_metric_th=10.0)
    sim = FogScenePerceptionSimulator(nusc_scene,
                                      pipeline,
                                      eval_metric=sim_args['eval_metric'],
                                      fog_density=sim_args['fog_density'],
                                      scatter_fraction=sim_args['scatter_fraction'],
                                      eval_k=sim_args['eval_k'],
                                      eval_metric_th=sim_args['eval_metric_th'])

    init_pred_tokens = list(sim.pred_candidate_info.keys())
    
    #print(len(init_pred_tokens))
    #print(sim.init_preds)
    # print(len(sim.passing_pred_tokens))

    ast_sim = PerceptionSimWrapper(sim)



    s0 = [0]
    # actions = [tuple([i]) for i in range(ast_sim.simulator.horizon)]
    # actions = [[7938],
    #    [3190],
    #    [5608],
    #    [ 659],
    #    [4168],
    #    [1657],
    #    [2424],
    #    [2770]]

    # actions = [[7938],
    #         [3168],
    #         [8969],
    #         [3556],
    #         [1942],
    #         [6433],
    #         [4906],
    #         [9205],
    #         [2237],
    #         [5712],
    #         [8742]]

    # actions = [[9628],
    #    [2175],
    #    [3061],
    #    [7308],
    #    [6890],
    #    [1106],
    #    [6571],
    #    [9259],
    #    [7465],
    #    [3814],
    #    [4265],
    #    [ 285],
    #    [8836],
    #    [4051],
    #    [1596]]
    # actions = [[ 187],
    #    [ 480],
    #    [3466],
    #    [7110],
    #    [4683],
    #    [1436],
    #    [5485],
    #    [6489],
    #    [8821],
    #    [3557],
    #    [8754],
    #    [ 760],
    #    [  89],
    #    [8274],
    #    [5792],
    #    [7399],
    #    [8213],
    #    [3954],
    #    [7686]]

    actions  =[[ 572],
       [7917],
       [1665],
       [4955],
       [3535],
       [4335],
       [7606],
       [2097],
       [ 762],
       [6636],
       [3366],
       [9912],
       [ 276],
       [6729],
       [2827],
       [3307],
       [6170],
       [8539],
       [4307],
       [ 483],
       [7761],
       [2975],
       [9097]]

    #     actions = pickle.load(f)[0]

    print('Actions: {}'.format(actions))
    render_path = '/mnt/hdd/ford_ws/output/ast/plots/test/'
    for i in range(1):
        print('-----------------')
        ast_sim.reset(s0)
        for a in actions:
            ast_sim.closed_loop_step(a)
            reward_info = ast_sim.get_reward_info()
            #print(ast_sim.is_terminal())
            # print(ast_sim.is_goal())
            ast_sim.simulator._render_detection(render_path)
            ast_sim.simulator._render_prediction(render_path)
            #plt.sca(ax)
            #plt.show(block=False)

    print(ast_sim.simulator.cnt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--config', type=str, default='./configs/pipeline_config.yaml',
                        help='specify the pipeline config')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        pipeline_config = yaml.load(f)
    
    main(pipeline_config)