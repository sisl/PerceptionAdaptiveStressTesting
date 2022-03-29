import argparse
from tqdm import tqdm
from collections import defaultdict
import pickle
from pathlib import Path
import yaml
from nuscenes import NuScenes
from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from sequential_perception.datasets import NuScenesScene
from sequential_perception.nuscenes_utils import get_ordered_samples


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
    dataroot = '/mnt/hdd/data/sets/nuscenes/v1.0-trainval'
    version = 'v1.0-trainval'
    
    output_root = '/mnt/hdd/data/sets/nuscenes/v1.0-trainval/exported_infos'
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load NuScenes
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)

    # Load PCDetDataset
    logger = common_utils.create_logger()
    args, detection_config = parse_config()
    cfg = load_pcdet_config(detection_config)

    pcdet_dataset = NuScenesDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(cfg.DATA_CONFIG.DATA_PATH), logger=logger
    )

    scene_infos_map = defaultdict(list)
    scene_info_idxs = defaultdict(list)

    print('Collecting infos...')
    for j in tqdm(range(len(pcdet_dataset))):
        data_dict = pcdet_dataset[j]
        sample_token = data_dict['metadata']['token']
        scene_token = nusc.get('sample', sample_token)['scene_token']
        scene_info_idxs[scene_token].append(j)

    print('Saving infos...')
    scene_tokens_found = list(scene_info_idxs.keys())
    for j in tqdm(range(len(scene_tokens_found))):
        scene_token = scene_tokens_found[j]
     
        scene_pcdet_infos = [pcdet_dataset[idx] for idx in scene_info_idxs[scene_token]]
        assert len(scene_pcdet_infos) > 5
        fname = scene_token + '.pkl'

        with open(output_path / fname, 'wb') as f:
            pickle.dump(scene_pcdet_infos, f)