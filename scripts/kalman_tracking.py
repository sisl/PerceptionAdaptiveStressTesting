from sequential_perception.kalman_tracker import Tracker
import argparse
import yaml
import numpy as np
from easydict import EasyDict
import json
from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionBox 
from nuscenes.eval.tracking.data_classes import TrackingBox 
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.tracking.data_classes import TrackingConfig
from nuscenes.eval.tracking.evaluate import TrackingEval
from pyquaternion import Quaternion
from collections import defaultdict
from sequential_perception.constants import NUSCENES_TRACKING_NAMES



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
    cfg_file = "OpenPCDet/tools/cfgs/nuscenes_models/cbgs_pp_multihead.yaml"
    parser.add_argument('--data_path', type=str, default='/scratch/hdelecki/ford/data/sets/nuscenes',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default='~/ford_ws/OpenPCDet/models/pp_multihead_nds5823_updated.pth', help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(cfg_file, cfg)

    return args, cfg


def main():
    #args, cfg = parse_config()
    #logger = common_utils.create_logger()
    #logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    #fg.DATA_CONFIG.VERSION = 'v1.0-mini'
    # demo_dataset = NuScenesDataset(
    #     dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
    #     root_path=Path(args.data_path), logger=logger
    # )
    #logger.info(f'Total number of samples: \t{len(demo_dataset)}')


    nusc_dataroot = '/scratch/hdelecki/ford/data/sets/nuscenes/v1.0-mini'
    nusc_version = 'v1.0-mini'
    detection_results_path = '/scratch/hdelecki/ford/output/pointpillars_eval/results_nusc.json'
    output_path = '/scratch/hdelecki/ford/output/tracking/kalman/nusc_tracking_results.json'
    eval_tracking_path='/scratch/hdelecki/ford/output/tracking/kalman/nusc_tracking_metrics.json'

    TRACKING_NAMES = NUSCENES_TRACKING_NAMES
    #TRACKING_NAMES = ['pedestrian']

    nusc = NuScenes(dataroot=nusc_dataroot,
                    version=nusc_version,
                    verbose=True)

    with open(detection_results_path) as f:
        det_data = json.load(f)
    assert 'results' in det_data, 'Error: No field `results` in result file. Please note that the result format changed.' \
    'See https://www.nuscenes.org/object-detection for more information.'

    all_results = EvalBoxes.deserialize(det_data['results'], DetectionBox)
    det_meta = det_data['meta']

    print('meta: ', det_meta)
    print("Loaded results from {}. Found detections for {} samples.".format(detection_results_path, len(all_results.sample_tokens)))

    # Collect tokens for all scenes in results
    scene_tokens = []
    for sample_token in all_results.sample_tokens:
        scene_token = nusc.get('sample', sample_token)['scene_token']
        if scene_token not in scene_tokens:
            scene_tokens.append(scene_token)

    tracking_results = {}

    for scene_token in scene_tokens:
        current_sample_token = nusc.get('scene', scene_token)['first_sample_token']

        trackers = {tracking_name: Tracker(tracking_name) for tracking_name in TRACKING_NAMES}

        while current_sample_token != '':
            tracking_results[current_sample_token] = []

            # Form detection observation:  [x, y, z, angle, l, h, w]
            # Group observations by detection name
            dets = {tracking_name: [] for tracking_name in TRACKING_NAMES}
            info = {tracking_name: [] for tracking_name in TRACKING_NAMES}
            for box in all_results.boxes[current_sample_token]:
                if box.detection_name not in TRACKING_NAMES:
                    continue
                detection = box
                information = np.array([box.detection_score])
                dets[box.detection_name].append(detection)
                info[box.detection_name].append(information)


            # Update Tracker
            for tracking_name in TRACKING_NAMES:
                updated_tracks = trackers[tracking_name].update(dets[tracking_name])
                tracking_results[current_sample_token] += updated_tracks


            current_sample_token = nusc.get('sample', current_sample_token)['next']
            


    tracking_meta = {
        "use_camera":   False,
        "use_lidar":    True,
        "use_radar":    False,
        "use_map":      False,
        "use_external": False,
    }

    tracking_output_data = {'meta': tracking_meta, 'results': tracking_results}

    with open(output_path, 'w') as outfile:
        json.dump(tracking_output_data, outfile)


if __name__ == '__main__':
    main()