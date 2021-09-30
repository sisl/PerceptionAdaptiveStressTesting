import argparse
import yaml
from pathlib import Path

#import mayavi.mlab as mlab
import numpy as np
from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset
from pcdet.datasets.nuscenes import nuscenes_utils
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

#from sequential_perception.detection import OpenPCDetector


def main():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--config', type=str, default='../configs/detection_config.yaml',
                        help='specify the config for detection')
    args = parser.parse_args()


    return


if __name__ == '__main__':
    main()