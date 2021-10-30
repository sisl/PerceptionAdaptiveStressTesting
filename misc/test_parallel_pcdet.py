



from nuscenes.eval.common.data_classes import EvalBoxes
import yaml
from pathlib import Path
import matplotlib.pyplot as plt

from nuscenes import NuScenes
from nuscenes.eval.detection.data_classes import DetectionBox

from pcdet.config import cfg_from_yaml_file, cfg
from pcdet.utils import common_utils
from pcdet.models import build_network, load_data_to_gpu

from sequential_perception.detectors import PCDetModule
from sequential_perception.datasets import NuScenesScene, PCDetSceneData
from sequential_perception.nuscenes_utils import get_ordered_samples, load_sample_gt, vis_sample_pointcloud

# from memory_profiler import profile


def load_config():
    #parser = argparse.ArgumentParser(description='arg parser')
    # parser.add_argument('--config', type=str, default='/mnt/hdd/hdelecki/ford_ws/src/SequentialPerceptionPipeline/configs/detection_config.yaml',
    #                     help='specify the config for detection')
    #args = parser.parse_args()
    cfg_path = Path('/mnt/hdd/hdelecki/ford_ws/src/SequentialPerceptionPipeline/configs/detection_config.yaml')
    with open(cfg_path, 'r') as f:
        detection_config = yaml.load(f)

    return detection_config


def load_pcdet_config(detection_config):
    model_config_path = detection_config['MODEL_CONFIG']
    pcdet_config = cfg_from_yaml_file(model_config_path, cfg)
    if 'train' in detection_config['NUSCENES_SPLIT']:
        pcdet_config['DATA_CONFIG']['INFO_PATH']['test'] = pcdet_config['DATA_CONFIG']['INFO_PATH']['train']

    return pcdet_config


def get_all_scenes_sample_tokens(nusc):
    scene_samples = {}
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_sample_tokens = get_ordered_samples(nusc, scene_token)
        scene_samples[scene_token] = scene_sample_tokens

    return scene_samples


def make_scene():
    dataroot = '/mnt/hdd/data/sets/nuscenes/v1.0-mini'
    version = 'v1.0-mini'
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)

    scene_token = nusc.scene[0]['token']
    nusc_scene = NuScenesScene(dataroot, nusc, scene_token)
    del nusc
    return nusc_scene

# @profile
def main():
    # load detection config
    config = load_config()
    cfg = load_pcdet_config(config)

    # load datasets
    dataroot = '/mnt/hdd/data/sets/nuscenes/v1.0-mini'
    version = 'v1.0-mini'
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)

    scene_token = nusc.scene[1]['token']
    nusc_scene = NuScenesScene(dataroot, nusc, scene_token)


    scene_sample_tokens = get_ordered_samples(nusc, scene_token)
    logger = common_utils.create_logger()
    pcdet_scene = PCDetSceneData(scene_sample_tokens,dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(cfg.DATA_CONFIG.DATA_PATH), logger=logger)


    # build model
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=pcdet_scene)
    model.load_params_from_file(filename=config['MODEL_CKPT'], logger=logger, to_cpu=True)

    # built module
    detector = PCDetModule(model, nusc_scene)

    # run detection for scene (or a sample)
    data_dict = pcdet_scene[13]
    sample_token = data_dict['metadata']['token']
    detections = detector([data_dict])

    # plot detections
    gt_boxes = load_sample_gt(nusc_scene, sample_token, DetectionBox)
    det_boxes = EvalBoxes.deserialize(detections['results'], DetectionBox)

    ax = vis_sample_pointcloud(nusc_scene,
                                sample_token,
                                gt_boxes=gt_boxes,
                                pred_boxes=det_boxes,
                                pc=data_dict['points'][:, :5].T,
                                savepath=None)
    plt.sca(ax)
    plt.show()

if __name__ == '__main__':
    main()