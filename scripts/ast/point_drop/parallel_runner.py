import pickle
from multiprocessing import Pool
import yaml
from runner_mcts_point_drop import runner as mcts_runner



def run(pipeline_args):
    # Overall settings
    max_path_length = 7
    s_0 = [0]
    base_log_dir = './fde10_motion'
    # experiment settings
    run_experiment_args = {'snapshot_mode': 'last',
                           'snapshot_gap': 1,
                           'log_dir': None,
                           'exp_name': None,
                           'seed': 0,
                           'n_parallel': 1,
                           'tabular_log_file': 'progress.csv'
                           }

    # runner settings
    runner_args = {'n_epochs': 100,
                   'batch_size': 100,
                   'plot': False
                   }

    # env settings
    env_args = {'id': 'ast_toolbox:Perception',
                'blackbox_sim_state': True,
                'open_loop': False,
                'fixed_init_state': True,
                's_0': s_0,
                }

    sim_args = {'blackbox_sim_state': True,
                'open_loop': False,
                'fixed_initial_state': True,
                'max_path_length': max_path_length,
                }

    # reward settings
    reward_args = {'use_heuristic': False}

    # spaces settings
    spaces_args = {'rsg_length': 4}

    # MCTS Settings ----------------------------------------------------------------------
    mcts_type = 'mcts'

    mcts_sampler_args = {}

    mcts_algo_args = {'max_path_length': max_path_length,
                        'stress_test_mode': 2,
                        'ec': 100.0,
                        'n_itr': 50,
                        'k': 0.5,
                        'alpha': 0.5,
                        'clear_nodes': True,
                        'log_interval': 50,
                        'plot_tree': False,
                        'plot_path': None,
                        'log_dir': None,
                        }

    mcts_bpq_args = {'N': 10}

    # MCTS settings
    run_experiment_args['log_dir'] = base_log_dir + '/mcts'
    run_experiment_args['exp_name'] = 'mcts'

    mcts_algo_args['max_path_length'] = max_path_length
    mcts_algo_args['log_dir'] = run_experiment_args['log_dir']
    mcts_algo_args['plot_path'] = run_experiment_args['log_dir']

    mcts_runner(
        mcts_type=mcts_type,
        env_args=env_args,
        run_experiment_args=run_experiment_args,
        sim_args=sim_args,
        perception_args=pipeline_args,
        reward_args=reward_args,
        spaces_args=spaces_args,
        algo_args=mcts_algo_args,
        runner_args=runner_args,
        bpq_args=mcts_bpq_args,
        sampler_args=mcts_sampler_args,
        save_expert_trajectory=True,
    )

if __name__ == '__main__':



    pipeline_config_path = '/mnt/hdd/hdelecki/ford_ws/src/SequentialPerceptionPipeline/configs/pipeline_config.yaml'
    with open(pipeline_config_path, 'r') as f:
        pipeline_config = yaml.load(f, yaml.SafeLoader)


    scene_token = 'fcbccedd61424f1b85dcbf8f897f9754'
    target_instance = '045cd82a77a1472499e8c15100cb5ff3'
    metric = 'MinFDEK'
    k = 1
    threshold = 20
    drop_likelihood = 0.1
    max_range = 35
    pipeline_args = {'pipeline_config': pipeline_config,
                     'scene_token': scene_token,
                     'target_instance': target_instance,
                     'drop_likelihood': drop_likelihood,
                     'eval_metric': metric,
                     'eval_k': k,
                     'eval_metric_th': threshold,
                     'no_pred_failure': False,
                     'max_range': max_range
                    }

    pipelines = 4*[pipeline_args]

    pool = Pool(8)
    pool.map(run, pipelines) 


