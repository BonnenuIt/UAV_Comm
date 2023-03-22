import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import numpy as np

class myenvEnv(gym.Env):
    def __init__(self, size=10, ActionSize=2):
        self.dim = 3
        self.min_pos = np.array([-size, -size, -size])  # The minimum position of the point
        self.max_pos = np.array([size, size, size])  # The maximum position of the point
        self.min_action = np.array([-ActionSize, -ActionSize, -ActionSize])  # The minimum action
        self.max_action = np.array([ActionSize, ActionSize, ActionSize])  # The maximum action
        
        self.action_space = spaces.Box(low=self.min_action, high=self.max_action, shape=(self.dim,), dtype=np.float)  # The continuous action space
        
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(low=self.min_pos, high=self.max_pos, shape=(self.dim,), dtype=np.float),  # The continuous observation space
                "target": spaces.Box(low=self.min_pos, high=self.max_pos, shape=(self.dim,), dtype=np.float),
            }
        )

        self._agent_location = None  # The current position of the point
        self._target_location = None  # The starting position of the point
        
        self.reward_range = (-np.inf, np.inf)  # The range of possible reward values
        
    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {"distance": np.linalg.norm(self._agent_location - self._target_location), 
                "_agent_location": self._agent_location,
                "_target_location": self._target_location, 
                "NumOfStep": self.NumOfStep}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.NumOfStep = 0
        self.total_reward = 0
        # Reset the environment to its initial state and return the initial observation
        self._target_location = np.random.uniform(self.min_pos, self.max_pos, size=3)
        self._agent_location = self._target_location.copy()
        
        observation = self._get_obs()
        info = self._get_info()

        return observation, info
        
    def step(self, action):
        # Take a step in the environment with the given action and return the next observation, reward, terminated flag, and info dictionary
        action = np.clip(action, self.min_action, self.max_action)
        previous_action = self._agent_location.copy()
        self._agent_location = np.clip(self._agent_location + action, self.min_pos, self.max_pos)
        reward = np.linalg.norm(self._agent_location - previous_action)
        terminated = False
        self.total_reward = self.total_reward + reward
        self.NumOfStep = self.NumOfStep + 1

        if reward < 2 and self.NumOfStep > 1:
            reward = 10
            terminated = True
            self.last_episode = {"r": self.total_reward + reward, "l": self.NumOfStep}

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info

    def render(self, mode='human'):
        # Render the environment
        pass
        
    def close(self):
        # Clean up any resources used by the environment
        pass

def myenvEnv_register():
    register(
        id='myenv-v0',
        entry_point='myenv:myenvEnv',
        # max_episode_steps=100,
    )