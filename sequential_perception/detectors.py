from typing import Dict, List
from pathlib import Path
from nuscenes.nuscenes import NuScenes
import torch
from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset
from pcdet.datasets.nuscenes import nuscenes_utils
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from sequential_perception.datasets import PCDetNuScenesDataset, PCDetSceneData
from sequential_perception.nuscenes_utils import get_ordered_samples

def load_pcdet_config(detection_config):
    model_config_path = detection_config['MODEL_CONFIG']
    pcdet_config = cfg_from_yaml_file(model_config_path, cfg)
    if 'train' in detection_config['NUSCENES_SPLIT']:
        pcdet_config['DATA_CONFIG']['INFO_PATH']['test'] = pcdet_config['DATA_CONFIG']['INFO_PATH']['train']

    return pcdet_config


class OpenPCDetector:
    def __init__(self, 
                 model_config_path: str,
                 ckpt_path: str,
                 nuscenes: NuScenes,
                 pcdet_infos: NuScenesDataset,
                 ext: str = '.pkl'):

        config = cfg_from_yaml_file(model_config_path, cfg)
        #self.data_path = data_path
        self.ckpt_path = ckpt_path
        self.ext = ext
        self.nuscenes = nuscenes
        self.pcdet_dataset = pcdet_infos
        
        model = build_network(model_cfg=config.MODEL, num_class=len(config.CLASS_NAMES), dataset=self.pcdet_dataset)
        model.load_params_from_file(filename=ckpt_path, logger=pcdet_infos.logger, to_cpu=False)
        model.cuda()
        model.eval()
        self.model = model
        return


    @torch.no_grad()
    def __call__(self, data_dicts: List[Dict]) -> Dict:
        pred_dict_batch = []
        batch_dict = {'metadata': [], 'frame_id':[]}
        for data_dict in data_dicts:
            data_dict = self.pcdet_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = self.model.forward(data_dict)

            pred_dict_batch += pred_dicts
            batch_dict['metadata'].append(data_dict['metadata'][0])
            batch_dict['frame_id'].append(data_dict['frame_id'][0])

        annos = self.pcdet_dataset.generate_prediction_dicts(batch_dict, pred_dict_batch, self.pcdet_dataset.class_names)
 
        nusc_annos = nuscenes_utils.transform_det_annos_to_nusc_annos(annos, self.nuscenes)
        nusc_annos['meta'] = {
            'use_camera': False,
            'use_lidar': True,
            'use_radar': False,
            'use_map': False,
            'use_external': False,
        }    
        return nusc_annos


class PCDetModule:
    def __init__(self,
                 model,
                 nuscenes: NuScenes,
                 ext: str = '.pkl'):
        model.cuda()
        model.eval()
        self.model = model
        self.ext = ext
        self.nuscenes = nuscenes
        self.pcdet_dataset = model.dataset

    @torch.no_grad()
    def __call__(self, data_dicts: List[Dict]) -> Dict:
        pred_dict_batch = []
        batch_dict = {'metadata': [], 'frame_id':[]}
        for data_dict in data_dicts:
            data_dict = self.pcdet_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = self.model.forward(data_dict)

            pred_dict_batch += pred_dicts
            batch_dict['metadata'].append(data_dict['metadata'][0])
            batch_dict['frame_id'].append(data_dict['frame_id'][0])

        annos = self.pcdet_dataset.generate_prediction_dicts(batch_dict, pred_dict_batch, self.pcdet_dataset.class_names)
 
        nusc_annos = nuscenes_utils.transform_det_annos_to_nusc_annos(annos, self.nuscenes)
        nusc_annos['meta'] = {
            'use_camera': False,
            'use_lidar': True,
            'use_radar': False,
            'use_map': False,
            'use_external': False,
        }    

        return nusc_annos