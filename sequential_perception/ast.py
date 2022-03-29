import numpy as np
from gym.spaces.discrete import Discrete
from gym.spaces.tuple import Tuple
from ast_toolbox.simulators import ASTSimulator
from ast_toolbox.spaces import ASTSpaces
from ast_toolbox.rewards import ASTReward
from nuscenes.nuscenes import NuScenes
from sequential_perception.simulators import ClosedLoopPointDropPerceptionSimulator as  PerceptionSim
from sequential_perception.classical_pipeline import ClassicalPerceptionPipeline


class PerceptionASTSimulator(ASTSimulator):
    def __init__(self,
                 nuscenes: NuScenes,
                 pipeline: ClassicalPerceptionPipeline,
                 perception_args = None,
                 **kwargs):
                 
        if perception_args is None:
            perception_args = {}

        self.simulator = PerceptionSim(nuscenes, pipeline, **perception_args)

        super().__init__(**kwargs)

    def simulate(self, actions, s_0):
        action_list = [a[0] for a in actions]
        return self.simulator.simulate(action_list, s_0[0]) 

    def reset(self, s_0):
        super(PerceptionASTSimulator, self).reset(s_0=s_0)
        self.observation = np.array([self.simulator.reset(s_0[0])])
        return self.observation_return()

    def closed_loop_step(self, action):
        self.observation = np.array([self.simulator.step_simulation(action[0])])
        return self.observation_return()

    def get_reward_info(self):
        reward_info = {'action_prob': self.simulator.action_prob,
                       'is_terminal': self.is_terminal(),
                       'is_goal': self.is_goal()
                       }
        return reward_info

    def is_goal(self):
        return self.simulator.is_goal()
    
    def is_terminal(self):
        return self.simulator.is_terminal()

    def clone_state(self):
        return [self.simulator.step]

    def restore_state(self, in_simulator_state):
        pass

    def render(self, **kwargs):
        return self.simulator.render()


class PerceptionSimWrapper(ASTSimulator):
    def __init__(self,
                 simulator,
                 **kwargs):
        self.simulator = simulator
        super().__init__(**kwargs)

    def simulate(self, actions, s_0):
        action_list = [a[0] for a in actions]
        return self.simulator.simulate(action_list, s_0[0]) 

    def reset(self, s_0):
        super(PerceptionSimWrapper, self).reset(s_0=s_0)
        self.observation = np.array([self.simulator.reset(s_0[0])])
        return self.observation_return()

    def closed_loop_step(self, action):
        self.observation = np.array([self.simulator.step_simulation(action[0])])
        return self.observation_return()

    def get_reward_info(self):
        reward_info = {'log_action_prob': self.simulator.log_action_prob,
                'is_terminal': self.is_terminal(),
                'is_goal': self.is_goal()
                }
        return reward_info

    def is_goal(self):
        return self.simulator.is_goal()
    
    def is_terminal(self):
        return self.simulator.is_terminal()

    def clone_state(self):
        return [self.simulator.step]

    def restore_state(self, in_simulator_state):
        pass

    def render(self, **kwargs):
        pass


class PerceptionASTSpaces(ASTSpaces):
    def __init__(self, rsg_length=3):
        self.rsg_length = rsg_length
        super().__init__()

    @property
    def action_space(self):
        return Tuple([Discrete(10**self.rsg_length-1)])

    @property
    def observation_space(self):
        return Tuple([Discrete(10**self.rsg_length-1)])


class PerceptionASTReward(ASTReward):
    def __init__(self, use_heuristic=False, expect_log=True):
        super().__init__()
        self.use_heuristic = use_heuristic
        self.expect_log = expect_log

    def give_reward(self, action, **kwargs):

        info = kwargs['info']
        is_goal = info["is_goal"]
        is_terminal = info["is_terminal"]

        if is_goal:  # We found a failure
            reward = 0
        elif is_terminal:
            # Heuristic reward tbd
            if self.use_heuristic:
                heuristic_reward = 0
            else:
                # No Herusitic
                heuristic_reward = 0
            reward = -1000000 - 10000 * heuristic_reward
        else:

            if self.expect_log:
                log_action_prob = info['log_action_prob']
                reward = 1e-3*log_action_prob
            else:
                action_prob = info['action_prob']
                if action_prob < 1e-200:
                    action_prob = 1e-200
                reward = np.log(action_prob)  # No failure or horizon yet
        
        if reward > 0:
            print('Something weird')
            print(info)

        return reward


