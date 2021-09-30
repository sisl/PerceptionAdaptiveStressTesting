from typing import Dict, List
from pathlib import Path
from nuscenes.nuscenes import NuScenes
import torch
from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset
from pcdet.datasets.nuscenes import nuscenes_utils
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from sequential_perception.datasets import PCDetNuScenesDataset

def load_pcdet_config(detection_config):
    model_config_path = detection_config['MODEL_CONFIG']
    pcdet_config = cfg_from_yaml_file(model_config_path, cfg)
    if 'train' in detection_config['NUSCENES_SPLIT']:
        pcdet_config['DATA_CONFIG']['INFO_PATH']['test'] = pcdet_config['DATA_CONFIG']['INFO_PATH']['train']

    return pcdet_config


# class OpenPCDetector:
#     def __init__(self, 
#                  detection_config: Dict,
#                  data_path: str,
#                  ckpt_path: str,
#                  ext: str = '.pkl'):
class OpenPCDetector:
    def __init__(self, 
                 model_config_path: str,
                 ckpt_path: str,
                 nuscenes: NuScenes,
                 pcdet_infos: NuScenesDataset,
                 #pcdet_nusc_dataset: PCDetNuScenesDataset,
                 #nuscenes: NuScenes,
                 ext: str = '.pkl'):

        #self.detection_config = detection_config
        #self.model_config = 
        #cfg = load_pcdet_config(detection_config)
        config = cfg_from_yaml_file(model_config_path, cfg)
        #self.data_path = data_path
        self.ckpt_path = ckpt_path
        self.ext = ext
        self.nuscenes = nuscenes
        #self.nuscenes = pcdet_nusc_dataset.nuscenes

        
        #logger = common_utils.create_logger()
        #dataset = NuScenesDataset(dataset_cfg=config.DATA_CONFIG, class_names=config.CLASS_NAMES, training=False,
                                #   root_path=Path(data_path), logger=logger)

        #self.pcdet_dataset = dataset
        #self.pcdet_dataset = pcdet_nusc_dataset.pcdet_infos
        self.pcdet_dataset = pcdet_infos

        # self.token_to_idx_map = self._get_reverse_sample_map()
        
        model = build_network(model_cfg=config.MODEL, num_class=len(config.CLASS_NAMES), dataset=self.pcdet_dataset)
        model.load_params_from_file(filename=ckpt_path, logger=pcdet_infos.logger, to_cpu=False)
        model.cuda()
        model.eval()
        self.model = model
        return

    # def _get_reverse_sample_map(self):
    #     reverse_map = {}
    #     for idx, data_dict in enumerate(self.pcdet_dataset):
    #         sample_token = data_dict['metadata']['token']
    #         reverse_map[sample_token] = idx
    #     return reverse_map

    # def get_data_for_sample(self, sample_token: str):
    #     idx = self.token_to_idx_map[sample_token]
    #     data_dict = self.pcdet_dataset[idx]
    #     return data_dict

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

        # NOTE: Neet to convert these OpenPCDet annotations to NuScenes detections  with the following
        # TODO: Do conversions here?
 
        #nusc = NuScenes(version=detection_config['NUSCENES_VERSION'], dataroot=detection_config['NUSCENES_DATAROOT'], verbose=True)
        nusc_annos = nuscenes_utils.transform_det_annos_to_nusc_annos(annos, self.nuscenes)
        nusc_annos['meta'] = {
            'use_camera': False,
            'use_lidar': True,
            'use_radar': False,
            'use_map': False,
            'use_external': False,
        }    
        #return annos
        return nusc_annos