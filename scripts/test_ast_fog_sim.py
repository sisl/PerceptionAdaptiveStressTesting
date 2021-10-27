import pickle
import numpy as np
from sequential_perception.ast import FogPerceptionASTSimulator
import yaml
from nuscenes import NuScenes
from sequential_perception.classical_pipeline import build_pipeline
import torch
import random
import time


if __name__ == '__main__':

    pipeline_config_path = '/mnt/hdd/hdelecki/ford_ws/src/SequentialPerceptionPipeline/configs/pipeline_config.yaml'

    np.random.seed(0)
    torch.random.manual_seed(0)
    random.seed(0)

    
    with open(pipeline_config_path, 'r') as f:
        pipeline_config = yaml.load(f, yaml.SafeLoader)


    def build_simulator(pipeline_config, perception_args, sim_args):
            # Create NuScenes
        # pipeline_config = perception_args['pipeline_config']
        nuscenes = NuScenes(version=pipeline_config['NUSCENES_VERSION'],
                            dataroot=pipeline_config['NUSCENES_DATAROOT'],
                            verbose=True)

        # Create Pipeline
        pipeline = build_pipeline(pipeline_config, nuscenes)

        # Create Simulator
        sim = FogPerceptionASTSimulator(nuscenes, pipeline, perception_args=perception_args, **sim_args)

        return sim
    
    sim_args = {}

    scene_token = 'fcbccedd61424f1b85dcbf8f897f9754'
    target_instance = '045cd82a77a1472499e8c15100cb5ff3'
    metric = 'MinFDEK'
    k = 5
    threshold = 20
    scatter_fraction = 0.05
    fog_density = 0.1
    max_range = 35
    perception_args = {'pipeline_config': pipeline_config,
                     'scene_token': scene_token,
                     'target_instance': target_instance,
                     'scatter_fraction': scatter_fraction,
                     'fog_density': fog_density,
                     'eval_metric': metric,
                     'eval_k': k,
                     'eval_metric_th': threshold,
                     'no_pred_failure': False,
                     'max_range': max_range
                    }

    simulator = build_simulator(pipeline_config, perception_args, sim_args)
    print(simulator.simulator.horizon)
    
    s0 = [0]
    actions = [tuple([i]) for i in range(simulator.simulator.horizon)]
    # actions = [(26,), (78,), (79,), (38,), (37,), (0,), (0,), (0,), (0,), (0,)]
    #actions = [(987,), (966,), (137,), (185,), (479,)] # from 0.25 + adek
    # actions = [(825,), (455,), (562,), (506,), (877,)] # from 0.1 + adek
    #actions = [(381,), (814,), (189,), (170,), (115,)] # 4.0 adek thresh + 0.1 like
    # actions =  [(488,), (950,), (300,), (260,), (497,)] # 4.0 adek + 0.1 and new bounding boxes
    # actions = [(426,), (375,), (547,), (340,), (246,)] # 4.0 adek + 0.01
    # actions = [(6706,), (5684,), (5779,), (3095,), (5842,)] # 10.0 fdek + 0.1

    # fname = './scripts/ast/point_drop/' + metric + '-' + str(k) + '-'+ str(threshold) + '-' + str(drop_likelihood) + '/mcts/best_actions.p'
    # with open(fname, 'rb') as f:
    #     actions = pickle.load(f)[0]

    print('Actions: {}'.format(actions))

    # for i in range(6):

    print('-----------------')
    start = time.time()
    simulator.reset(s0)
    for a in actions:
        simulator.closed_loop_step(a)
        reward_info = simulator.get_reward_info()
        print(simulator.is_terminal())
    duration = time.time() - start
    print('Total time: {} sec'.format(duration))
    print('Total time: {} sec'.format((duration/len(actions))))

        # print('render')
        # simulator.render()