import pickle
import fire
import json
from garage.experiment.local_runner import LocalRunner
import numpy as np
import torch
from pathlib import Path
import tensorflow as tf
from garage.experiment import run_experiment, SnapshotConfig
from garage.tf.experiment import LocalTFRunner

import ast_toolbox.mcts.BoundedPriorityQueues as BPQ
from ast_toolbox.algos import MCTS
from ast_toolbox.algos import MCTSBV
from ast_toolbox.algos import MCTSRS

# Import the AST classes
from ast_toolbox.envs import ASTEnv
from ast_toolbox.samplers import ASTVectorizedSampler
from ast_toolbox.samplers import BatchSampler

from nuscenes.nuscenes import NuScenes
from sequential_perception.classical_pipeline import build_pipeline
from sequential_perception.ast import PerceptionASTSimulator, PerceptionASTSpaces, PerceptionASTReward, PerceptionSimWrapper



def runner(
    simulator,
    mcts_type=None,
    env_args=None,
    run_experiment_args=None,
    sim_args=None,
    #perception_args=None,
    reward_args=None,
    spaces_args=None,
    algo_args=None,
    runner_args=None,
    bpq_args=None,
    sampler_args=None,
    save_expert_trajectory=False,
    
    # log_dir='.',
):
    if mcts_type is None:
        mcts_type = 'mcts'

    if env_args is None:
        env_args = {}

    if run_experiment_args is None:
        run_experiment_args = {}

    # if sim_args is None:
    #     sim_args = {}
    if sim_args is None:
        sim_args = {}

    # if perception_args is None:
    #     raise NotImplementedError()

    if reward_args is None:
        reward_args = {}

    if spaces_args is None:
        spaces_args = {}

    if algo_args is None:
        algo_args = {}

    if runner_args is None:
        runner_args = {}

    if sampler_args is None:
        sampler_args = {}

    if bpq_args is None:
        bpq_args = {}

    if 'n_parallel' in run_experiment_args:
        n_parallel = run_experiment_args['n_parallel']
    else:
        n_parallel = 1
        run_experiment_args['n_parallel'] = n_parallel

    if 'max_path_length' in sim_args:
        max_path_length = sim_args['max_path_length']
    else:
        max_path_length = 5
        sim_args['max_path_length'] = max_path_length

    if 'batch_size' in runner_args:
        batch_size = runner_args['batch_size']
    else:
        batch_size = max_path_length * n_parallel
        runner_args['batch_size'] = batch_size

    logpath = Path(run_experiment_args['log_dir'])
    logpath.mkdir(parents=True, exist_ok=True)

    def run_task():

        seed = 0
        # top_k = 10
        np.random.seed(seed)

        # Create log dir
        snapshot_config = SnapshotConfig('./', 'none', 1)
        runner = LocalRunner(snapshot_config, max_cpus=1)

        # Instantiate AST classes
        sim = PerceptionSimWrapper(simulator, **sim_args)
        reward_function = PerceptionASTReward(**reward_args)
        spaces = PerceptionASTSpaces(**spaces_args)

        # Create the environment
        if 'id' in env_args:
            env_args.pop('id')
        env = ASTEnv(simulator=sim,
                        reward_function=reward_function,
                        spaces=spaces,
                        **env_args
                        )
        print('Open loop')
        print(env.open_loop)

        top_paths = BPQ.BoundedPriorityQueue(**bpq_args)

        if mcts_type == 'mcts':
            print('mcts')
            algo = MCTS(env=env,
                        top_paths=top_paths,
                        **algo_args)
        elif mcts_type == 'mctsbv':
            print('mctsbv')
            algo = MCTSBV(env=env,
                            top_paths=top_paths,
                            **algo_args)
        elif mcts_type == 'mctsrs':
            print('mctsrs')
            algo = MCTSRS(env=env,
                            top_paths=top_paths,
                            **algo_args)
        else:
            raise NotImplementedError

        sampler_cls = ASTVectorizedSampler
        runner.setup(algo=algo,
                            env=env,
                            sampler_cls=sampler_cls,
                            sampler_args=sampler_args)

        # Run the experiment
        runner.train(**runner_args)

        # Try to load best actions
        with open(Path(algo_args['log_dir']) / 'best_actions.p', 'rb') as f:
            all_best_actions = pickle.load(f)

        if len(all_best_actions) > 0:
            best_actions = all_best_actions[-1]
            
            # Create plot path
            plot_path = algo_args['log_dir'] + '/plots/'
            Path(plot_path).mkdir(parents=True, exist_ok=True)

            sim.reset([0])
            reward = 0.0
            for action in best_actions:
                sim.step(action)
                if sim.simulator.step > 0:
                    sim.simulator._render_detection(plot_path)
                    sim.simulator._render_prediction(plot_path)
                goal = sim.is_goal()
                reward += np.log(sim.simulator.action_prob)
            summary = sim.simulator.failure_log

        else:
            summary = sim.simulator.failure_log

        # Write summary to json
        summary_path = Path(algo_args['log_dir']) / 'summary.pkl'
        with open(summary_path, 'wb') as f:
            #json.dump(summary, f)
            pickle.dump(summary, f)

        data_path = Path(algo_args['log_dir']) / 'data.pkl'
        with open(data_path, 'wb') as f:
            pickle.dump(sim.simulator.failure_perception_data, f)

    run_task()


if __name__ == '__main__':
    fire.Fire()