from nuscenes.eval.tracking.data_classes import TrackingBox 
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.tracking.data_classes import TrackingConfig
from nuscenes.eval.tracking.evaluate import TrackingEval
from pyquaternion import Quaternion
from sequential_perception.tracking_render import CustomTrackingRenderer

def main():
    # TODO add argparse

    nusc_dataroot = '/scratch/hdelecki/ford/data/sets/nuscenes/v1.0-mini'
    nusc_version = 'v1.0-mini'
    
    # For heuristic
    # tracking_results_path = '/scratch/hdelecki/ford/output/nusc_tracking_results.json'
    # eval_tracking_path='/scratch/hdelecki/ford/output/nusc_tracking_evaluation/heuristic_tracker_mini'

    # For kalman
    tracking_results_path = '/scratch/hdelecki/ford/output/tracking/kalman/nusc_tracking_results.json'
    eval_tracking_path='/scratch/hdelecki/ford/output/tracking/kalman/nusc_tracking_evaluation/kalman_tracker_mini'

    render_classes = None
    
    
    tracking_eval_cfg = config_factory('tracking_nips_2019')
    eval_set = 'mini_val'

    nusc_tracking_eval = TrackingEval(config=tracking_eval_cfg, result_path=tracking_results_path, 
                                      eval_set=eval_set, nusc_version=nusc_version,
                                      output_dir=eval_tracking_path, nusc_dataroot=nusc_dataroot,
                                      render_classes=render_classes, verbose=True)
    nusc_tracking_eval.main(render_curves=0)

if __name__ == '__main__':
    main()