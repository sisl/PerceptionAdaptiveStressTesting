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


def main():
    logger = common_utils.create_logger()
    #logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
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
    print(model.module_list)

    model.cuda()
    model.eval()

    data_dict = demo_dataset[10]
    data_dict = demo_dataset.collate_batch([data_dict])
    n_modules = 2
    print(data_dict.keys())
    print(data_dict['metadata'])
    print(data_dict['voxel_coords'].shape)
    print(data_dict['voxels'].shape)
    # for i in range(n_modules):
    #     cur_module = model.module_list[i]
    #     batch_dict = cur_module(batch_dict)
    #     print(batch_dict.keys())
    load_data_to_gpu(data_dict)
    with torch.no_grad():
        for i in range(n_modules):
            cur_module = model.module_list[i]
            data_dict = cur_module(data_dict)
            print(data_dict.keys())


    pillar_features = data_dict['pillar_features'].cpu()
    spatial_features = data_dict['spatial_features'].cpu()

    print(pillar_features.shape)
    print(spatial_features.shape)
    print(torch.max(spatial_features))

    import matplotlib.pyplot as plt
    import numpy as np

    # fig, axs = plt.subplots(8,8, figsize=(15, 6), facecolor='w', edgecolor='k')
    # fig, axs = plt.subplots(8,8, figsize=(15, 15), facecolor='w', edgecolor='k')
    # # fig.subplots_adjust(hspace = .5, wspace=.001)

    # axs = axs.ravel()
    # for i in range(64):
    #     axs[i].imshow(spatial_features[0, i, :, :])
    #     #axs[i].set_title(str(i))

    for j in range(0,64,4):
        fig, axs = plt.subplots(2,2, facecolor='w',edgecolor='k')
        axs = axs.ravel()
        for i in range(4):
            axs[i].imshow(spatial_features[0, i+j, :, :])
        

    plt.show()




    # model.cuda()
    # model.eval()
    # start = time.time()
    # with torch.no_grad():
    #     for idx, data_dict in enumerate(demo_dataset):
    #         logger.info(f'Detection sample index: \t{idx + 1}')
    #         data_dict = demo_dataset.collate_batch([data_dict])

    #         load_data_to_gpu(data_dict)
    #         pred_dicts, _ = model.forward(data_dict)

    #         pred_dict_batch += pred_dicts
    #         batch_dict['metadata'].append(data_dict['metadata'][0])
    #         batch_dict['frame_id'].append(data_dict['frame_id'][0])



if __name__ == '__main__':
    main()