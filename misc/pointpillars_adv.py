import argparse
#import glob
from pathlib import Path
import json
from nuscenes.nuscenes import NuScenes
import yaml
from pathlib import Path
import time

import numpy as np
from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset
from pcdet.datasets.nuscenes import nuscenes_utils
import torch
import torch.nn.functional as F

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


def get_pfn_latent(pfn, inputs):
    if inputs.shape[0] > pfn.part:
        # nn.Linear performs randomly when batch size is too large
        num_parts = inputs.shape[0] // pfn.part
        part_linear_out = [pfn.linear(inputs[num_part*pfn.part:(num_part+1)*pfn.part])
                            for num_part in range(num_parts+1)]
        x = torch.cat(part_linear_out, dim=0)
    else:
        x = pfn.linear(inputs)
    torch.backends.cudnn.enabled = False
    x = pfn.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if pfn.use_norm else x
    torch.backends.cudnn.enabled = True
    x = F.relu(x)
    x_max = torch.max(x, dim=1, keepdim=True)[0]

    x_argmax = torch.argmax(x, dim=1)

    if pfn.last_vfe:
        return x_max, x_argmax
    else:
        x_repeat = x_max.repeat(1, inputs.shape[1], 1)
        x_concatenated = torch.cat([x, x_repeat], dim=2)


def get_pointpillars_critical_set(model, batch_dict):
    pillar_vfe = model.module_list[0]
    assert len(pillar_vfe.pfn_layers) == 1
    pfn_layer = pillar_vfe.pfn_layers[0]


    voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
    points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
    f_cluster = voxel_features[:, :, :3] - points_mean

    f_center = torch.zeros_like(voxel_features[:, :, :3])
    f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * pillar_vfe.voxel_x + pillar_vfe.x_offset)
    f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * pillar_vfe.voxel_y + pillar_vfe.y_offset)
    f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * pillar_vfe.voxel_z + pillar_vfe.z_offset)

    if pillar_vfe.use_absolute_xyz:
        features = [voxel_features, f_cluster, f_center]
    else:
        features = [voxel_features[..., 3:], f_cluster, f_center]

    if pillar_vfe.with_distance:
        points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
        features.append(points_dist)
    features = torch.cat(features, dim=-1)

    voxel_count = features.shape[1]
    mask = pillar_vfe.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
    mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
    features *= mask

    latent, latent_idxs = get_pfn_latent(pfn_layer, features)


    critical_point_idxs = {}
    for i in range(latent_idxs.shape[0]):
        critical_point_idxs[i] = torch.unique(latent_idxs[i, :])

    critical_sets = {}
    #points = batch_dict['points'].cpu()
    for pillar_idx, point_idxs in critical_point_idxs.items():
        pillar_critical_set = voxel_features[pillar_idx, point_idxs, :].cpu()
        critical_sets[pillar_idx] = pillar_critical_set


    return critical_sets
    # for pfn in self.pfn_layers:
    #     features = pfn(features)






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
    # with torch.no_grad():
    #     for i in range(n_modules):
    #         cur_module = model.module_list[i]
    #         data_dict = cur_module(data_dict)
    #         print(data_dict.keys())


    # pillar_features = data_dict['pillar_features'].cpu()
    # spatial_features = data_dict['spatial_features'].cpu()

    # print(pillar_features.shape)
    # print(spatial_features.shape)
    # print(torch.max(spatial_features))

    # import matplotlib.pyplot as plt
    # import numpy as np

    with torch.no_grad():
        critical_sets = get_pointpillars_critical_set(model, data_dict)


    # create array of points 
    critical_set_array = np.vstack(list(critical_sets.values()))

    critical_set_array = critical_set_array[~np.all(critical_set_array == 0, axis=1)]
    critical_set_array  = np.unique(critical_set_array , axis=0)
    print(critical_set_array.shape)
    print(data_dict['points'].shape)


    # Half of the points (~110000) are in the critical set (!!)

    # Need to reduce to points near target

    print(critical_set_array[:, :20])
    print(data_dict['points'][:, :20])

    print(critical_set_array[0, 0] in data_dict['points'])

    for i in range(data_dict['points'].shape[0]):
        row = data_dict['points'][i, :]
        if critical_set_array[0, 0] in row:
            print(row)
            break






    #list_of_points = list(critical_sets.values()
    #print(len(list_of_points))
    #critical_set_points = torch.Tensor(list(critical_sets.values()))


    

    # fig, axs = plt.subplots(8,8, figsize=(15, 6), facecolor='w', edgecolor='k')
    # fig, axs = plt.subplots(8,8, figsize=(15, 15), facecolor='w', edgecolor='k')
    # # fig.subplots_adjust(hspace = .5, wspace=.001)

    # axs = axs.ravel()
    # for i in range(64):
    #     axs[i].imshow(spatial_features[0, i, :, :])
    #     #axs[i].set_title(str(i))

    # for j in range(0,64,4):
    #     fig, axs = plt.subplots(2,2, facecolor='w',edgecolor='k')
    #     axs = axs.ravel()
    #     for i in range(4):
    #         axs[i].imshow(spatial_features[0, i+j, :, :])
        

    # plt.show()




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