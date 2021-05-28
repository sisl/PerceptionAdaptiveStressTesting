from sequential_perception.heuristic_tracker import HeuristicTracker
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
    output_path = '/scratch/hdelecki/ford/output/nusc_tracking_results.json'
    eval_tracking_path='/scratch/hdelecki/ford/output/nusc_tracking_metrics.json'



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

        trackers = {tracking_name: HeuristicTracker(tracking_name) for tracking_name in NUSCENES_TRACKING_NAMES}

        while current_sample_token != '':
            tracking_results[current_sample_token] = []

            # Form detection observation:  [x, y, z, angle, l, h, w]
            # Group observations by detection name
            dets = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}
            info = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}
            for box in all_results.boxes[current_sample_token]:
                if box.detection_name not in NUSCENES_TRACKING_NAMES:
                    continue
                #q = Quaternion(box.rotation)
                # angle = q.angle if q.axis[2] > 0 else -q.angle

                # detection = np.array([box.translation[0],  box.translation[1], box.translation[2],
                #                     angle,
                #                     box.size[1], box.size[0], box.size[2]])
                #print('detection: ', detection)
                detection = box
                information = np.array([box.detection_score])
                dets[box.detection_name].append(detection)
                info[box.detection_name].append(information)

            for tracking_name in NUSCENES_TRACKING_NAMES:
                if len(dets[tracking_name]) > 0:
                    
                    current_tracks = trackers[tracking_name].update(dets[tracking_name])


                    for i in range(len(current_tracks)):
                        track_box = current_tracks[i]['track']
                        track_id = current_tracks[i]['track_id']

                        translation = track_box.translation


                        sample_result = {
                            'sample_token': sample_token,
                            'translation': track_box.translation,
                            'size': track_box.size,
                            'rotation': track_box.rotation,
                            'velocity': [0, 0],
                            'tracking_id': str(int(track_id)),
                            'tracking_name': tracking_name,
                            'tracking_score': track_box.detection_score
                        }




                        #sample_result = format_sample_result(current_sample_token, tracking_name, trackers[i])
                        tracking_results[current_sample_token].append(sample_result)

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


    # tracking_eval_cfg = config_factory('tracking_nips_2019')
    # eval_set = 'mini_val'
    # nusc_tracking_eval = TrackingEval(config=tracking_eval_cfg, result_path=output_path, 
    #                                   eval_set=eval_set, nusc_version=nusc_version,
    #                                   output_dir=eval_tracking_path, nusc_dataroot=nusc_dataroot, verbose=True)
    # nusc_tracking_eval.main(render_curves=1)


    




    # with torch.no_grad():
    #     for idx, data_dict in enumerate(demo_dataset):
    #         logger.info(f'Visualized sample index: \t{idx + 1}')
    #         data_dict = demo_dataset.collate_batch([data_dict])

    #         load_data_to_gpu(data_dict)
    #         pred_dicts, _ = model.forward(data_dict)

    #         pred_dict_batch += pred_dicts
    #         batch_dict['metadata'].append(data_dict['metadata'][0])
    #         batch_dict['frame_id'].append(data_dict['frame_id'][0])

            
    #         sample_token = data_dict['metadata']
    #         scene_sample_tokens.append(sample_token)
    #         sample = nusc.get('sample', sample_token)
            
    #         # get ego pose + yaw, give to tracker
    #         ego_pose = nusc.get('ego_pose', sample['data']['LIDAR_TOP'])
    #         x = ego_pose['translation'][0]
    #         y = ego_pose['translation'][1]
    #         yaw = quaternion_yaw(Quaternion(ego_pose['rotation']))
    #         tracker.getPoseOffset(x, y, yaw)

    #         # update tracker
    #         track_history = tracker.updateHistory(pred_dicts[0])
    #         trackInfo = tracker.getTrackingInfo()
    #         if trackInfo:
    #             for track in track_history:
    #                 if len(track)==tracker.currentTime:
    #                     detection = track[-1]
    #                     translation = detection.location
    #                     size = detection.boxDims
    #                     rotation = 
                        
    #         #if at end of scene, format tracking results and reset trckers
    #         if sample['next'] == '':
    #             #dump tracks
    #             agent = AgentBox()
    #             tracker = Tracker()





    # annos = demo_dataset.generate_prediction_dicts(batch_dict, pred_dict_batch, demo_dataset.class_names)
    # print(annos)

    # result_str, result_dict = demo_dataset.evaluation(annos, demo_dataset.class_names, output_path='/scratch/hdelecki/ford/output')
    # print(result_str)

    #             offset = nusc.
    #         tracker.getOffset(demo_dataset.getLatLon(idx))
    #         track_history = tracker.updateHistory(pred_dicts[0])
    #         trackInfo = tracker.getTrackingInfo()
    #         if trackInfo:
    #             img = agent.getBoxes(pred_dicts[0], None, trackInfo, tracker.currentTime)
    #             plt.clf()
    #             plt.imshow(img)
    #             plt.savefig("InputImages/objectDetect{:04d}.png".format(idx))

    #logger.info('Demo done.')


if __name__ == '__main__':
    main()