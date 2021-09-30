from collections import defaultdict
from typing import Any, Callable, Dict, List
from nuscenes import NuScenes
import numpy as np
from nuscenes.eval.common.config import config_factory

from nuscenes.eval.common.data_classes import EvalBox, EvalBoxType, EvalBoxes
from nuscenes.eval.common.utils import DetectionBox, center_distance, scale_iou, yaw_diff, velocity_l2, attr_acc, cummean
from nuscenes.eval.detection.data_classes import DetectionMetricData
from nuscenes.eval.prediction.config import PredictionConfig, load_prediction_config
from nuscenes.eval.tracking.utils import category_to_tracking_name
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.eval.detection.data_classes import DetectionBox, DetectionConfig
from nuscenes.eval.tracking.data_classes import TrackingBox, TrackingConfig
from nuscenes.eval.prediction.data_classes import Prediction
from nuscenes.prediction.helper import PredictHelper

class DetectionEvaluation:
    def __init__(self):
        return

def load_sample_gt(nusc: NuScenes, sample_token: str, box_cls, verbose: bool = False) -> EvalBoxes:
    """
    Loads ground truth boxes from DB.
    :param nusc: A NuScenes instance.
    :param eval_split: The evaluation split for which we load GT boxes.
    :param box_cls: Type of box to load, e.g. DetectionBox or TrackingBox.
    :param verbose: Whether to print messages to stdout.
    :return: The GT boxes.
    """
    # Init.
    if box_cls == DetectionBox:
        attribute_map = {a['token']: a['name'] for a in nusc.attribute}

    # if verbose:
    #     print('Loading annotations for {} split from nuScenes version: {}'.format(eval_split, nusc.version))
    # Read out all sample_tokens in DB.
    #sample_tokens_all = [s['token'] for s in nusc.sample]
    #assert len(sample_tokens_all) > 0, "Error: Database has no samples!"

    # Only keep samples from this split.
    #splits = create_splits_scenes()

    # Check compatibility of split with nusc_version.
    version = nusc.version
    # if eval_split in {'train', 'val', 'train_detect', 'train_track'}:
    #     assert version.endswith('trainval'), \
    #         'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
    # elif eval_split in {'mini_train', 'mini_val'}:
    #     assert version.endswith('mini'), \
    #         'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
    # elif eval_split == 'test':
    #     assert version.endswith('test'), \
    #         'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
    # else:
    #     raise ValueError('Error: Requested split {} which this function cannot map to the correct NuScenes version.'
    #                      .format(eval_split))

    # if eval_split == 'test':
    #     # Check that you aren't trying to cheat :).
    #     assert len(nusc.sample_annotation) > 0, \
    #         'Error: You are trying to evaluate on the test set but you do not have the annotations!'

    # sample_tokens = []
    # for sample_token in sample_tokens_all:
    #     scene_token = nusc.get('sample', sample_token)['scene_token']
    #     scene_record = nusc.get('scene', scene_token)
    #     if scene_record['name'] in splits[eval_split]:
    #         sample_tokens.append(sample_token)

    all_annotations = EvalBoxes()

    # Load annotations and filter predictions and annotations.
    tracking_id_set = set()
    #for sample_token in tqdm.tqdm(sample_tokens, leave=verbose):

    sample = nusc.get('sample', sample_token)
    sample_annotation_tokens = sample['anns']

    sample_boxes = []
    for sample_annotation_token in sample_annotation_tokens:

        sample_annotation = nusc.get('sample_annotation', sample_annotation_token)
        if box_cls == DetectionBox:
            # Get label name in detection task and filter unused labels.
            detection_name = category_to_detection_name(sample_annotation['category_name'])
            if detection_name is None:
                continue

            # Get attribute_name.
            attr_tokens = sample_annotation['attribute_tokens']
            attr_count = len(attr_tokens)
            if attr_count == 0:
                attribute_name = ''
            elif attr_count == 1:
                attribute_name = attribute_map[attr_tokens[0]]
            else:
                raise Exception('Error: GT annotations must not have more than one attribute!')

            sample_boxes.append(
                box_cls(
                    sample_token=sample_token,
                    translation=sample_annotation['translation'],
                    size=sample_annotation['size'],
                    rotation=sample_annotation['rotation'],
                    velocity=nusc.box_velocity(sample_annotation['token'])[:2],
                    num_pts=sample_annotation['num_lidar_pts'] + sample_annotation['num_radar_pts'],
                    detection_name=detection_name,
                    detection_score=-1.0,  # GT samples do not have a score.
                    attribute_name=attribute_name
                )
            )
        elif box_cls == TrackingBox:
            # Use nuScenes token as tracking id.
            tracking_id = sample_annotation['instance_token']
            tracking_id_set.add(tracking_id)

            # Get label name in detection task and filter unused labels.
            # Import locally to avoid errors when motmetrics package is not installed.
            
            tracking_name = category_to_tracking_name(sample_annotation['category_name'])
            if tracking_name is None:
                continue

            sample_boxes.append(
                box_cls(
                    sample_token=sample_token,
                    translation=sample_annotation['translation'],
                    size=sample_annotation['size'],
                    rotation=sample_annotation['rotation'],
                    velocity=nusc.box_velocity(sample_annotation['token'])[:2],
                    num_pts=sample_annotation['num_lidar_pts'] + sample_annotation['num_radar_pts'],
                    tracking_id=tracking_id,
                    tracking_name=tracking_name,
                    tracking_score=-1.0  # GT samples do not have a score.
                )
            )
        else:
            raise NotImplementedError('Error: Invalid box_cls %s!' % box_cls)

    all_annotations.add_boxes(sample_token, sample_boxes)

    if verbose:
        print("Loaded ground truth annotations for {} samples.".format(len(all_annotations.sample_tokens)))

    return all_annotations

class SampleDetectionMetrics:
    def __init__(self):
        return

# class SamplePredictionMetrics:

#class Scene

class SampleMetricData:
    def __init__(self, sample_token: str) -> None:
        self.sample_token = sample_token
        self.sample_data = defaultdict(lambda: defaultdict(np.nan))
        return

    def add_metric(self, metric: str, instance_token: str, metric_value: float):
        self.sample_data[metric][instance_token] = metric_value


class SceneMetricData:
    def __init__(self, scene_token: str) -> None:
        self.scene_token = scene_token
        #self.scene_data = defaultdict(lambda: defaultdict(list))
        return
    
    def add_sample(self, sample: SampleMetricData):
        # for metric in sample.sample_data.keys():
        #     for inst, metric_val in sample.sample_data[metric]:
        self.scene_data[sample.sample_token] = sample.sample_data


class PipelineEvaluation:
    def __init__(self, nuscenes: NuScenes, class_name: str) -> None:
        assert class_name in ['car', 'truck', 'bus']
        self.nuscenes = nuscenes
        self.class_name = class_name
        return

    def accumulate_samples(self):
        pass

    def accumulate_scenes(self):
        pass

    def evaluate(self, detections: EvalBoxes, tracks: EvalBoxes, predictions: List[Prediction]):
        # For each sample, gather all detections, tracks, predictions
        # assert detections.sample_tokens == tracks.sample_tokens
        
        sample_tokens = detections.sample_tokens

        detection_data = defaultdict(dict)
        track_data = defaultdict(dict)
        pred_data = defaultdict(dict)

        helper = PredictHelper(self.nuscenes)
        config = load_prediction_config(helper)
        for sample_token in sample_tokens:
            # Load GT boxes
            gt_boxes = load_sample_gt(self.nuscenes, sample_token, TrackingBox)

            sample_detections = detections[sample_token]
            sample_detection_metrics = accumulate_sample_detections(gt_boxes[sample_token],
                                                                    sample_detections,
                                                                    self.class_name,
                                                                    center_distance,
                                                                    2.0,
                                                                    verbose=True)

            detection_data[sample_token] = sample_detection_metrics

            sample_tracking_metrics = {}

            sample_predictions = [p for p in predictions if p.sample == sample_token]
            #sample_predictions = defaultdict(lambda: defaultdict(lambda: np.nan))
            sample_prediction_metrics = compute_prediction_metrics(sample_predictions, helper, config)
            pred_data[sample_token] = sample_prediction_metrics


        # agent_failure_metrics[instance]['MinADEK'].append(aggs['MinADEK']['RowMean'][-1])
        # agent_failure_metrics[instance]['MinFDEK'].append(aggs['MinFDEK']['RowMean'][-1])
        # agent_failure_metrics[instance]['MissRateTopK_2'].append(aggs['MissRateTopK_2']['RowMean'][-1])


            #sample_prediction_metrics = sample_prediction_metrics()


        return detection_data, track_data, pred_data

    def evaluate_predictions(self, predictions: List[Prediction]):
        # For each sample, gather all detections, tracks, predictions
        # assert detections.sample_tokens == tracks.sample_tokens

        pred_data = defaultdict(dict)

        helper = PredictHelper(self.nuscenes)
        config = load_prediction_config(helper)
        # for sample_token in sample_tokens:
        #     # Load GT boxes
        #     gt_boxes = load_sample_gt(self.nuscenes, sample_token, TrackingBox)

        #     sample_detections = detections[sample_token]
        #     sample_detection_metrics = accumulate_sample_detections(gt_boxes[sample_token],
        #                                                             sample_detections,
        #                                                             self.class_name,
        #                                                             center_distance,
        #                                                             2.0,
        #                                                             verbose=True)

        #     detection_data[sample_token] = sample_detection_metrics

        #     sample_tracking_metrics = {}

        #     sample_predictions = [p for p in predictions if p.sample == sample_token]
            #sample_predictions = defaultdict(lambda: defaultdict(lambda: np.nan))
        prediction_metrics = compute_prediction_metrics(predictions, helper, config)
        #pred_data[sample_token] = sample_prediction_metrics
        return prediction_metrics


        # agent_failure_metrics[instance]['MinADEK'].append(aggs['MinADEK']['RowMean'][-1])
        # agent_failure_metrics[instance]['MinFDEK'].append(aggs['MinFDEK']['RowMean'][-1])
        # agent_failure_metrics[instance]['MissRateTopK_2'].append(aggs['MissRateTopK_2']['RowMean'][-1])


            #sample_prediction_metrics = sample_prediction_metrics()


        return pred_data





def accumulate_sample_detections(gt_boxes: List[EvalBoxType],
                                pred_boxes: List[EvalBoxType],
                                class_name: str,
                                dist_fcn: Callable,
                                dist_th: float,
                                verbose: bool = False):
    """
    Calculate per-instance error metrics for pred_boxes. Metrics are tran
    :param gt_boxes: List of predicted EvalBox types.
    :param pred_boxes: List of ground-truth EvalBox types.
    :return: dict[metric][instance] : float. Dictionary mapping metric name and instance token to value
    """
    # ---------------------------------------------
    # Organize input and initialize accumulators.
    # ---------------------------------------------

    # Count the positives.
    npos = len([1 for gt_box in gt_boxes if gt_box.tracking_name == class_name])
    if npos == 0:
        return defaultdict(lambda: defaultdict(np.nan))

    # Organize the predictions in a single list.

    pred_boxes_list = [box for box in pred_boxes if box.detection_name == class_name]
    pred_confs = [box.detection_score for box in pred_boxes_list]

    # if verbose:
    #     print("Found {} PRED of class {} out of {} total across {} samples.".
    #           format(len(pred_confs), class_name, len(pred_boxes), len(pred_boxes.sample_tokens)))

    # Sort by confidence.
    sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(pred_confs))][::-1]

    # Do the actual matching.
    tp = []  # Accumulator of true positives.
    fp = []  # Accumulator of false positives.
    conf = []  # Accumulator of confidences.

    # match_data holds the extra metrics we calculate for each match.
    match_data = {'trans_err': defaultdict(lambda: np.nan),
                  'vel_err': defaultdict(lambda: np.nan),
                  'scale_err': defaultdict(lambda: np.nan),
                  'orient_err': defaultdict(lambda: np.nan),
                  'attr_err': defaultdict(lambda: np.nan),
                  'conf': defaultdict(lambda: np.nan)}

    # ---------------------------------------------
    # Match and accumulate match data.
    # ---------------------------------------------

    taken = set()  # Initially no gt bounding box is matched.
    for ind in sortind:
        pred_box = pred_boxes_list[ind]
        min_dist = np.inf
        match_gt_idx = None

        for gt_idx, gt_box in enumerate(gt_boxes):

            # Find closest match among ground truth boxes
            if gt_box.tracking_name == class_name and not (pred_box.sample_token, gt_idx) in taken:
                this_distance = dist_fcn(gt_box, pred_box)
                if this_distance < min_dist:
                    min_dist = this_distance
                    match_gt_idx = gt_idx

        # If the closest match is close enough according to threshold we have a match!
        is_match = min_dist < dist_th

        if is_match:
            taken.add((pred_box.sample_token, match_gt_idx))

            #  Update tp, fp and confs.
            tp.append(1)
            fp.append(0)
            conf.append(pred_box.detection_score)

            # Since it is a match, update match data also.
            gt_box_match = gt_boxes[match_gt_idx]

            # GT tracking id == instance_token
            match_data['trans_err'][gt_box_match.tracking_id] = center_distance(gt_box_match, pred_box)
            match_data['vel_err'][gt_box_match.tracking_id] = velocity_l2(gt_box_match, pred_box)
            match_data['scale_err'][gt_box_match.tracking_id] = 1 - scale_iou(gt_box_match, pred_box)

            # Barrier orientation is only determined up to 180 degree. (For cones orientation is discarded later)
            period = np.pi if class_name == 'barrier' else 2 * np.pi
            match_data['orient_err'][gt_box_match.tracking_id] = yaw_diff(gt_box_match, pred_box, period=period)

            # match_data['attr_err'].append(1 - attr_acc(gt_box_match, pred_box))
            # match_data['conf'].append(pred_box.detection_score)

        else:
            # No match. Mark this as a false positive.
            tp.append(0)
            fp.append(1)
            conf.append(pred_box.detection_score)

    # Check if we have any matches. If not, just return a "no predictions" array.
    if np.sum(tp) == 0:
        return defaultdict(lambda: defaultdict(np.nan))
    # if len(match_data['trans_err']) == 0:
    #     pass
        #return DetectionMetricData.no_predictions()
    return match_data


def compute_prediction_metrics(predictions: List[Prediction],
                    helper: PredictHelper, config: PredictionConfig) -> Dict[str, Any]:
    """
    Computes metrics from a set of predictions.
    :param predictions: List of prediction JSON objects.
    :param helper: Instance of PredictHelper that wraps the nuScenes val set.
    :param config: Config file.
    :return: Metrics. Nested dictionary where keys are metric names and value is a dictionary
        mapping the Aggregator name to the results.
    """
    n_preds = len(predictions)
    # containers = {metric.name: np.zeros((n_preds, metric.shape)) for metric in config.metrics}
    prediction_data = defaultdict(lambda: defaultdict(lambda: np.nan))
    for i, prediction in enumerate(predictions):
        #prediction = Prediction.deserialize(prediction_str)
        ground_truth = helper.get_future_for_agent(prediction.instance, prediction.sample,
                                                   config.seconds, in_agent_frame=False)
        for metric in config.metrics:
            prediction_data[metric.name][prediction.instance] = metric(ground_truth, prediction)[-1]
    # aggregations: Dict[str, Dict[str, List[float]]] = defaultdict(dict)
    # for metric in config.metrics:
    #     for agg in metric.aggregators:
    #         aggregations[metric.name][agg.name] = agg(containers[metric.name])
    # return aggregations
    return prediction_data


def compute_prediction_metrics(prediction: Prediction,
                    ground_truth: np.ndarray, config: PredictionConfig) -> Dict[str, Any]:
    """
    Computes metrics from a set of predictions.
    :param predictions: List of prediction JSON objects.
    :param helper: Instance of PredictHelper that wraps the nuScenes val set.
    :param config: Config file.
    :return: Metrics. Nested dictionary where keys are metric names and value is a dictionary
        mapping the Aggregator name to the results.
    """
    #n_preds = len(predictions)
    # containers = {metric.name: np.zeros((n_preds, metric.shape)) for metric in config.metrics}
    prediction_data = defaultdict(lambda: defaultdict(lambda: np.nan))
    #for i, prediction in enumerate(predictions):
        #prediction = Prediction.deserialize(prediction_str)
        # ground_truth = helper.get_future_for_agent(prediction.instance, prediction.sample,
        #                                            config.seconds, in_agent_frame=False)
    for metric in config.metrics:
        prediction_data[metric.name][prediction.instance] = metric(ground_truth, prediction)
    # aggregations: Dict[str, Dict[str, List[float]]] = defaultdict(dict)
    # for metric in config.metrics:
    #     for agg in metric.aggregators:
    #         aggregations[metric.name][agg.name] = agg(containers[metric.name])
    # return aggregations
    return prediction_data


def accumulate(gt_boxes: EvalBoxes,
               pred_boxes: EvalBoxes,
               class_name: str,
               dist_fcn: Callable,
               dist_th: float,
               verbose: bool = False) -> DetectionMetricData:
    """
    Average Precision over predefined different recall thresholds for a single distance threshold.
    The recall/conf thresholds and other raw metrics will be used in secondary metrics.
    :param gt_boxes: Maps every sample_token to a list of its sample_annotations.
    :param pred_boxes: Maps every sample_token to a list of its sample_results.
    :param class_name: Class to compute AP on.
    :param dist_fcn: Distance function used to match detections and ground truths.
    :param dist_th: Distance threshold for a match.
    :param verbose: If true, print debug messages.
    :return: (average_prec, metrics). The average precision value and raw data for a number of metrics.
    """
    # ---------------------------------------------
    # Organize input and initialize accumulators.
    # ---------------------------------------------

    # Count the positives.
    npos = len([1 for gt_box in gt_boxes.all if gt_box.detection_name == class_name])
    if verbose:
        print("Found {} GT of class {} out of {} total across {} samples.".
              format(npos, class_name, len(gt_boxes.all), len(gt_boxes.sample_tokens)))

    # For missing classes in the GT, return a data structure corresponding to no predictions.
    if npos == 0:
        return DetectionMetricData.no_predictions()

    # Organize the predictions in a single list.
    pred_boxes_list = [box for box in pred_boxes.all if box.detection_name == class_name]
    pred_confs = [box.detection_score for box in pred_boxes_list]

    if verbose:
        print("Found {} PRED of class {} out of {} total across {} samples.".
              format(len(pred_confs), class_name, len(pred_boxes.all), len(pred_boxes.sample_tokens)))

    # Sort by confidence.
    sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(pred_confs))][::-1]

    # Do the actual matching.
    tp = []  # Accumulator of true positives.
    fp = []  # Accumulator of false positives.
    conf = []  # Accumulator of confidences.

    # match_data holds the extra metrics we calculate for each match.
    match_data = {'trans_err': [],
                  'vel_err': [],
                  'scale_err': [],
                  'orient_err': [],
                  'attr_err': [],
                  'conf': []}

    # ---------------------------------------------
    # Match and accumulate match data.
    # ---------------------------------------------

    taken = set()  # Initially no gt bounding box is matched.
    for ind in sortind:
        pred_box = pred_boxes_list[ind]
        min_dist = np.inf
        match_gt_idx = None

        for gt_idx, gt_box in enumerate(gt_boxes[pred_box.sample_token]):

            # Find closest match among ground truth boxes
            if gt_box.detection_name == class_name and not (pred_box.sample_token, gt_idx) in taken:
                this_distance = dist_fcn(gt_box, pred_box)
                if this_distance < min_dist:
                    min_dist = this_distance
                    match_gt_idx = gt_idx

        # If the closest match is close enough according to threshold we have a match!
        is_match = min_dist < dist_th

        if is_match:
            taken.add((pred_box.sample_token, match_gt_idx))

            #  Update tp, fp and confs.
            tp.append(1)
            fp.append(0)
            conf.append(pred_box.detection_score)

            # Since it is a match, update match data also.
            gt_box_match = gt_boxes[pred_box.sample_token][match_gt_idx]

            match_data['trans_err'].append(center_distance(gt_box_match, pred_box))
            match_data['vel_err'].append(velocity_l2(gt_box_match, pred_box))
            match_data['scale_err'].append(1 - scale_iou(gt_box_match, pred_box))

            # Barrier orientation is only determined up to 180 degree. (For cones orientation is discarded later)
            period = np.pi if class_name == 'barrier' else 2 * np.pi
            match_data['orient_err'].append(yaw_diff(gt_box_match, pred_box, period=period))

            match_data['attr_err'].append(1 - attr_acc(gt_box_match, pred_box))
            match_data['conf'].append(pred_box.detection_score)

        else:
            # No match. Mark this as a false positive.
            tp.append(0)
            fp.append(1)
            conf.append(pred_box.detection_score)

    # Check if we have any matches. If not, just return a "no predictions" array.
    if len(match_data['trans_err']) == 0:
        return DetectionMetricData.no_predictions()

    # ---------------------------------------------
    # Calculate and interpolate precision and recall
    # ---------------------------------------------

    # Accumulate.
    tp = np.cumsum(tp).astype(float)
    fp = np.cumsum(fp).astype(float)
    conf = np.array(conf)

    # Calculate precision and recall.
    prec = tp / (fp + tp)
    rec = tp / float(npos)

    rec_interp = np.linspace(0, 1, DetectionMetricData.nelem)  # 101 steps, from 0% to 100% recall.
    prec = np.interp(rec_interp, rec, prec, right=0)
    conf = np.interp(rec_interp, rec, conf, right=0)
    rec = rec_interp

    # ---------------------------------------------
    # Re-sample the match-data to match, prec, recall and conf.
    # ---------------------------------------------

    for key in match_data.keys():
        if key == "conf":
            continue  # Confidence is used as reference to align with fp and tp. So skip in this step.

        else:
            # For each match_data, we first calculate the accumulated mean.
            tmp = cummean(np.array(match_data[key]))

            # Then interpolate based on the confidences. (Note reversing since np.interp needs increasing arrays)
            match_data[key] = np.interp(conf[::-1], match_data['conf'][::-1], tmp[::-1])[::-1]

    # ---------------------------------------------
    # Done. Instantiate MetricData and return
    # ---------------------------------------------
    return DetectionMetricData(recall=rec,
                               precision=prec,
                               confidence=conf,
                               trans_err=match_data['trans_err'],
                               vel_err=match_data['vel_err'],
                               scale_err=match_data['scale_err'],
                               orient_err=match_data['orient_err'],
                               attr_err=match_data['attr_err'])