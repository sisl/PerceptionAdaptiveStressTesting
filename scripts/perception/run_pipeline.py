import argparse
import time
import yaml
import json
from nuscenes import NuScenes
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from sequential_perception.classical_pipeline import ClassicalPerceptionPipeline
from sequential_perception.evaluation import PipelineEvaluation

from pathlib import Path
from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset # as PCDetNuScenesDataset
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils


def main(pipeline_config):
    
    # Load PCDet config
    model_config_path = pipeline_config['DETECTION']['MODEL_CONFIG']
    pcdet_config = cfg_from_yaml_file(model_config_path, cfg)
    if 'train' in pipeline_config['NUSCENES_SPLIT']:
        pcdet_config['DATA_CONFIG']['INFO_PATH']['test'] = pcdet_config['DATA_CONFIG']['INFO_PATH']['train']
    config = cfg_from_yaml_file(model_config_path, cfg)
    
    # Create PCDet info dataset
    data_path = pipeline_config['NUSCENES_DATAROOT']
    pcdet_data_path = data_path.strip(pipeline_config['NUSCENES_VERSION'])
    logger = common_utils.create_logger()
    pcdet_infos = NuScenesDataset(dataset_cfg=config.DATA_CONFIG, class_names=config.CLASS_NAMES, training=False,
                                   root_path=Path(pcdet_data_path), logger=logger)

    # Create NuScenes
    nuscenes = NuScenes(version=pipeline_config['NUSCENES_VERSION'],
                        dataroot=pipeline_config['NUSCENES_DATAROOT'],
                        verbose=True)

    # Create Pipeline
    pipeline = ClassicalPerceptionPipeline(pipeline_config, nuscenes, pcdet_infos)

    ## Detection
    all_data_dicts = [data_dict for data_dict in pcdet_infos]
    start = time.time()
    print('Running Detection ...')
    detections = pipeline.run_detection(all_data_dicts)
    print('Done. Detection took {} sec'.format(time.time() - start))

    detection_results_path = Path(pipeline_config['DETECTION']['RESULTS'])
    detection_results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pipeline_config['DETECTION']['RESULTS'], 'w') as f:
        json.dump(detections, f)


    ## Tracking
    start = time.time()
    print('Running Tracking ...')
    tracks = pipeline.run_tracking(detections)
    print('Done. Tracking took {} sec'.format(time.time() - start))

    tracking_results_path = Path(pipeline_config['TRACKING']['RESULTS'])
    tracking_results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pipeline_config['TRACKING']['RESULTS'], 'w') as f:
        json.dump(tracks, f)

    ## Prediction
    print('Running Predicton ...')
    predict_tokens = get_prediction_challenge_split(pipeline_config['NUSCENES_SPLIT'], dataroot=pipeline_config['NUSCENES_DATAROOT'])

    start = time.time()
    preds = pipeline.run_prediction(tracks, predict_tokens)
    print('Done. Prediction took {} sec'.format(time.time() - start))
    print('Done')

    prediction_results_path = Path(pipeline_config['PREDICTION']['RESULTS'])
    prediction_results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pipeline_config['PREDICTION']['RESULTS'], 'w') as f:
        json.dump(preds, f)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--config', type=str, default='./configs/pipeline_config.yaml',
                        help='specify the pipeline config')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        pipeline_config = yaml.load(f)
    
    main(pipeline_config)
