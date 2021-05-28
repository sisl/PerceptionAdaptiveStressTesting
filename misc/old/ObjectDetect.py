import argparse
import yaml
import numpy as np
import torch
import glob
import matplotlib.pyplot as plt
from pathlib import Path
from easydict import EasyDict
from pcdet.utils import common_utils
from pcdet.config import cfg
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from Agent import AgentBox
from Tracking import Tracker
import os
import pickle
class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list
    def getLatLon(self, index):
        i = str(self.root_path).rfind("velodyne")
        gpsPath = str(self.root_path)[:i] + "oxts/data/"
        fileName = str(self.sample_file_list[index]).split("/")[-1][:-4] 
        fileName = gpsPath + fileName + ".txt"
        with open(fileName, "r") as reader:
            line = reader.readline()
            line = line.split(" ")
            return (float(line[0]), float(line[1]), float(line[5]))

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

def merge_new_config(config, new_config):
    if '_BASE_CONFIG_' in new_config:
        with open('OpenPCDet/tools/' + new_config['_BASE_CONFIG_'], 'r') as f:
            try:
                yaml_config = yaml.load(f, Loader=yaml.FullLoader)
            except:
                yaml_config = yaml.load(f)
        config.update(EasyDict(yaml_config))

    for key, val in new_config.items():
        if not isinstance(val, dict):
            config[key] = val
            continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)

    return config

def cfg_from_yaml_file(cfg_file, config):
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)

        merge_new_config(config=config, new_config=new_config)

    return config

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    cfg_file = "OpenPCDet/tools/cfgs/kitti_models/pv_rcnn.yaml"
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(cfg_file, cfg)

    return args, cfg

def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    ckpt = "OpenPCDet/Models/pv_rcnn_8369.pth"
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    agent = AgentBox()
    tracker = Tracker()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            #colorFunction = tracker.getColorFunction(pred_dicts[0])
            tracker.getOffset(demo_dataset.getLatLon(idx))
            track_history = tracker.updateHistory(pred_dicts[0])
            trackInfo = tracker.getTrackingInfo()
            if trackInfo:
                img = agent.getBoxes(pred_dicts[0], None, trackInfo, tracker.currentTime)
                plt.clf()
                plt.imshow(img)
                plt.savefig("InputImages/objectDetect{:04d}.png".format(idx))
    with open("trackingInfo_3.pkl", "wb") as handle:
        pickle.dump(tracker, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info('Demo done.')

if __name__ == '__main__':
    main()
