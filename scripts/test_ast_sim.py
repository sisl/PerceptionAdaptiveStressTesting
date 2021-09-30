import pickle
import numpy as np
from sequential_perception.ast import PerceptionASTSimulator
import yaml
from nuscenes import NuScenes
from sequential_perception.classical_pipeline import build_pipeline
import torch
import random


if __name__ == '__main__':

    pipeline_config_path = '/home/hdelecki/ford_ws/SequentialPerceptionPipeline/configs/pipeline_config.yaml'

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
        sim = PerceptionASTSimulator(nuscenes, pipeline, perception_args=perception_args, **sim_args)

        return sim
    
    # scene_token = '6f83169d067343658251f72e1dd17dbc'
    # target_instance = '6a0d7a293cc34a84b9857a4fa38cd4a3'

    # scene_token = 'd25718445d89453381c659b9c8734939'
    # target_instance = '9091a580e27b4c7a96060d109ce7ef4c'
    sim_args = {}
    # perception_args = {'scene_token': scene_token,
    #                    'target_instance': target_instance,
    #                    'max_range': 40,
    #                    'drop_likelihood': 0.0}

    # scene_token ='c5224b9b454b4ded9b5d2d2634bbda8a'
    # target_instance = '88e4abb15ade4097a1689225c13399ec'

    scene_token ='de7d80a1f5fb4c3e82ce8a4f213b450a'
    target_instance = '6a4b9f7abafd44f38ff670c3064b13dc'
    metric = 'MinFDEK'
    k = 5
    threshold = 20
    drop_likelihood = 0.0
    max_range = 35
    perception_args = {'pipeline_config': pipeline_config,
                     'scene_token': scene_token,
                     'target_instance': target_instance,
                     'drop_likelihood': drop_likelihood,
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
    simulator.reset(s0)
    for a in actions:
        simulator.closed_loop_step(a)
        reward_info = simulator.get_reward_info()
        print(simulator.is_terminal())

        # print('render')
        # simulator.render()
