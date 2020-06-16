import gym, ray
from ray.rllib.agents import sac
import numpy as np
class ExcursionEnv(gym.Env):

    def __init__(self, env_config):
        self.T = env_config["excursion_time"]
        self.win_reward = env_config["win_reward"]
        self.lose_reward_scalar = env_config["lose_reward"]
        self.negative_reward = env_config["negative_reward"]
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(-self.T, self.T, (2, ), dtype=np.float32)

        self.x = 0.0
        self.t = 0.0

    def reset(self):
        self.x = 0.0
        self.t = 0.0

        return self._observation_calculation()

    def _observation_calculation(self):
        return [self.x, self.t]

    def _reward_calculation(self, current_x: int, current_t: int, next_x: int) -> float:
        reward = 0.0
        if next_x < 0:
            reward += self.negative_reward
        if current_t == self.T-1:
            if next_x == 0:
                reward += self.win_reward
            else:
                reward += self.lose_reward_scalar*np.abs(next_x)
        
        return reward

    def step(self, action):
        assert action in [0, 1], action
        step_change = 1.0 if action == 0 else 0.0
        
        reward = self._reward_calculation(self.x, self.t, self.x + step_change)

        self.x += step_change
        self.t += 1.0

        done = self.t >= self.T

        return self._observation_calculation(), reward, done, {}
        

