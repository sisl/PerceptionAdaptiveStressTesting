import copy
import pickle
from pathlib import Path
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset
from pcdet.datasets.nuscenes import nuscenes_utils
from pcdet.utils import common_utils
from tqdm import tqdm


def create_nuscenes_info(version, data_path, save_path, max_sweeps=10):
    data_path = data_path / version
    save_path = save_path / version

    assert version in ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise NotImplementedError

    nusc = NuScenes(version=version, dataroot=data_path, verbose=True)
    available_scenes = nuscenes_utils.get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]
    train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in train_scenes])
    val_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in val_scenes])

    print('%s: train scene(%d), val scene(%d)' % (version, len(train_scenes), len(val_scenes)))

    train_nusc_infos, val_nusc_infos = nuscenes_utils.fill_trainval_infos(
        data_path=data_path, nusc=nusc, train_scenes=train_scenes, val_scenes=val_scenes,
        test='test' in version, max_sweeps=max_sweeps
    )

    if version == 'v1.0-test':
        print('test sample: %d' % len(train_nusc_infos))
        with open(save_path / f'nuscenes_infos_{max_sweeps}sweeps_test.pkl', 'wb') as f:
            pickle.dump(train_nusc_infos, f)
    else:
        print('train sample: %d, val sample: %d' % (len(train_nusc_infos), len(val_nusc_infos)))
        with open(save_path / f'nuscenes_infos_{max_sweeps}sweeps_train.pkl', 'wb') as f:
            pickle.dump(train_nusc_infos, f)
        with open(save_path / f'nuscenes_infos_{max_sweeps}sweeps_val.pkl', 'wb') as f:
            pickle.dump(val_nusc_infos, f)


if __name__ == '__main__':
    import yaml
    import argparse
    from pathlib import Path
    from easydict import EasyDict

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='./OpenPCDet/tools/cfgs/dataset_configs/nuscenes_dataset.yaml', help='specify the config of dataset')
    parser.add_argument('--func', type=str, default='create_nuscenes_infos', help='')
    parser.add_argument('--version', type=str, default='v1.0-mini', help='')
    args = parser.parse_args()

    if args.func == 'create_nuscenes_infos':
        dataset_cfg = EasyDict(yaml.load(open(args.cfg_file)))
        ROOT_DIR = Path('/mnt/hdd/data/sets/nuscenes')
        dataset_cfg.VERSION = args.version
        create_nuscenes_info(
            version=dataset_cfg.VERSION,
            data_path=ROOT_DIR,
            save_path=ROOT_DIR,
            max_sweeps=dataset_cfg.MAX_SWEEPS,
        )

        nuscenes_dataset = NuScenesDataset(
            dataset_cfg=dataset_cfg, class_names=None,
            root_path=ROOT_DIR,
            logger=common_utils.create_logger(), training=True
        )
        nuscenes_dataset.create_groundtruth_database(max_sweeps=dataset_cfg.MAX_SWEEPS)