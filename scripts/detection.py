import argparse
#import glob
from pathlib import Path
import json
from nuscenes.nuscenes import NuScenes
import yaml
from pathlib import Path
import time

#import numpy as np
from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset
from pcdet.datasets.nuscenes import nuscenes_utils
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
#from visual_utils import visualize_utils as V


def free_data_from_gpu(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, torch.Tensor):
            continue
        if key in ['frame_id', 'metadata', 'calib', 'image_shape']:
            continue
        batch_dict[key] = val.cpu()


# def parse_config():
#     parser = argparse.ArgumentParser(description='arg parser')
#     parser.add_argument('--cfg_file', type=str, default='../OpenPCDet/tools/cfgs/nuscenes_models/cbgs_pp_multihead.yaml',
#                         help='specify the config for demo')
#     parser.add_argument('--data_path', type=str, default='/scratch/hdelecki/ford/data/sets/nuscenes/v1.0-mini',
#                         help='specify the point cloud data file or directory')
#     parser.add_argument('--ckpt', type=str, default=None, help='../OpenPCDet/models/pp_multihead_nds5823_updated.pth')
#     parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

#     args = parser.parse_args()

#     cfg_from_yaml_file(args.cfg_file, cfg)

#     return args, cfg


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--config', type=str, default='/home/hdelecki/ford_ws/SequentialPerceptionPipeline/configs/detection_config.yaml',
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
    #pcdet_config['DATA_CONFIG']['DATA_PATH'] = detection_config['NUSCENES_DATAROOT']
    #pcdet_config['DATA_CONFIG']['VERSION'] = detection_config['NUSCENES_VERSION']

    return pcdet_config

def main():
    #args, cfg = parse_config()
    logger = common_utils.create_logger()
    #logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    #cfg.DATA_CONFIG.VERSION = 'v1.0-mini'
    
    args, detection_config = parse_config()
    cfg = load_pcdet_config(detection_config)


    demo_dataset = NuScenesDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(cfg.DATA_CONFIG.DATA_PATH), logger=logger
    )
    #print(demo_dataset.class_names)
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    pred_dict_batch = []
    data_dict_list = []
    batch_dict = {'metadata': [], 'frame_id':[]}

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=detection_config['MODEL_CKPT'], logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    start = time.time()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Detection sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])

            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            pred_dict_batch += pred_dicts
            batch_dict['metadata'].append(data_dict['metadata'][0])
            batch_dict['frame_id'].append(data_dict['frame_id'][0])
            # print(data_dict.keys())
            # print(data_dict)


    annos = demo_dataset.generate_prediction_dicts(batch_dict, pred_dict_batch, demo_dataset.class_names)
    #print(annos)

    #result_str, result_dict = demo_dataset.evaluation(annos, demo_dataset.class_names, output_path='/scratch/hdelecki/ford/output/detection')
    #print(result_str)
   
    nusc = NuScenes(version=detection_config['NUSCENES_VERSION'], dataroot=detection_config['NUSCENES_DATAROOT'], verbose=True)
    nusc_annos = nuscenes_utils.transform_det_annos_to_nusc_annos(annos, nusc)
    nusc_annos['meta'] = {
        'use_camera': False,
        'use_lidar': True,
        'use_radar': False,
        'use_map': False,
        'use_external': False,
    }

    print(time.time() - start)

    with open(detection_config['DETECTION_RESULTS'], 'w') as f:
        json.dump(nusc_annos, f)


    logger.info('Done.')


if __name__ == '__main__':
    main()
