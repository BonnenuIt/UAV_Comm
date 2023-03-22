import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import numpy as np
from .MyRandomenv import MyRandomenv
from .MyLine import SensorHight
from .MySensors import MySensor
from .Quadrotor.myquadrotor import MyQuadrotor
from .PathPlanning import mymaputils

def myenvobsEnv_register():
    register(
        id='myenvobs-v0',
        entry_point='myobsenv.myenvobs:myenvobsEnv',
        # max_episode_steps=100,
    )

class myenvobsEnv(gym.Env):
    def __init__(self, size=20, ActionSize=2):
        self.dim = 3
        self.MaxNumObs = 10
        self.SensorNum = 10
        self.SensorMaxContent = 10
        self.min_pos = np.array([0, 0, 0])  # The minimum position of the point
        self.max_pos = np.array([size, size, size])  # The maximum position of the point
        
        self.min_action = np.array([-ActionSize, -ActionSize, -ActionSize])  # The minimum action
        self.max_action = np.array([ActionSize, ActionSize, ActionSize])  # The maximum action
        
        self.action_space = spaces.Box(low=self.min_action, high=self.max_action, shape=(self.dim,), dtype=np.float)  # The continuous action space
        
        self.observation_space = self._obs_def(self.min_pos, self.max_pos, self.dim, self.MaxNumObs, self.SensorNum, self.SensorMaxContent)

        self._agent_location = None  # The current position of the point
        self._target_location = None  # The starting position of the point
        
        self.reward_range = (-np.inf, np.inf)  # The range of possible reward values

        #### Random Env ####
        self.MyRandomenv = MyRandomenv()
        self.MyRandomenv.obstacles = self.MyRandomenv.ObsGen(self.MaxNumObs, self.min_pos, self.max_pos)

        
        print("myenvobsEnv")
        
    def _obs_def(self, min_pos, max_pos, dim, MaxNumObs, SensorNum, SensorMaxContent):
        min_sensor = np.concatenate((min_pos, np.array([0])))
        max_sensor = np.concatenate((max_pos, np.array([SensorMaxContent])))
        min_sensor = np.expand_dims(min_sensor, 0).repeat(SensorNum, axis=0)
        max_sensor = np.expand_dims(max_sensor, 0).repeat(SensorNum, axis=0)
        min_obs = np.concatenate((min_pos, min_pos))
        max_obs = np.concatenate((max_pos, max_pos))
        min_obs = np.expand_dims(min_obs, 0).repeat(MaxNumObs, axis=0)
        max_obs = np.expand_dims(max_obs, 0).repeat(MaxNumObs, axis=0)
        observation_space = spaces.Dict(
            {
                "agent": spaces.Box(low=min_pos, high=max_pos, shape=(dim,), dtype=np.float),  # The continuous observation space
                "target": spaces.Box(low=min_pos, high=max_pos, shape=(dim,), dtype=np.float),
                "obstacle": spaces.Box(low=min_obs, high=max_obs, shape=(MaxNumObs, dim * 2), dtype=np.float),
                "sensor": spaces.Box(low=min_sensor, high=max_sensor, shape=(SensorNum, dim + 1), dtype=np.float),
            }
        )
        return observation_space

    def _get_obs(self):
        sensor_obs = []
        for i in self.SensorObjList:
            sensor_obs.append(np.concatenate((i.Location, [i.TotalContent])))
        obstacle_obs = np.array(self.MyRandomenv.obstacles)
        ob = np.zeros((self.MaxNumObs - obstacle_obs.shape[0], obstacle_obs.shape[1]))
        obstacle_obs = np.concatenate((obstacle_obs, ob), axis=0)
        return {"agent": self._agent_location, 
                "target": self._target_location, 
                "obstacle": obstacle_obs, 
                "sensor": np.array(sensor_obs), }

    def _get_info(self):
        return {"distance": np.linalg.norm(self._agent_location - self._target_location), 
                "_agent_location": self._agent_location,
                "_target_location": self._target_location, 
                "NumOfStep": self.NumOfStep}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        SensorNodes = np.random.uniform(low=self.min_pos[:2], high=self.max_pos[:2], size=(self.SensorNum, 2)).astype(int).tolist()
        SensorContent = np.random.uniform(low=1, high=self.SensorMaxContent, size=(self.SensorNum)).tolist()   # sensor 信息内容大小
        SensorNodes = SensorHight(SensorNodes, self.MyRandomenv.obstacles, HightL=self.min_pos[2], HightH=self.max_pos[2] * 0.2)      # sensor如果初步设定到障碍物内，则放到障碍物顶上
        
        self.SensorObjList = []
        for i in range(len(SensorNodes)):
            self.SensorObjList.append(MySensor(index=i+1, Location=SensorNodes[i], 
                                               Content=SensorContent[i], 
                                               Bandwidth=self.MyRandomenv.Bandwidth, TxPower=self.MyRandomenv.TxPower, 
                                               NoisePower=self.MyRandomenv.NoisePower, CommAlpha=self.MyRandomenv.CommAlpha))
        self.mapobs = mymaputils.myMap(self.MyRandomenv.obstacles, SensorNodes, dim = self.dim)
        
        self.NumOfStep = 0
        self.total_reward = 0
        # Reset the environment to its initial state and return the initial observation
        self._target_location = np.random.uniform(self.min_pos, self.max_pos, size=3)
        self._agent_location = self._target_location.copy()

        self.MyDrone = MyQuadrotor(Location=self._agent_location, CommMode=self.MyRandomenv.CommMode, Velocity=None)
        
        observation = self._get_obs()
        info = self._get_info()

        return observation, info
        
    def _reward_cal(self, old_loc, new_loc):
        reward = 0
        CollisionDetect = self.mapobs.collision(old_loc, new_loc)
        if CollisionDetect == True:
            reward = -1
            return reward
        
        if np.linalg.norm(old_loc - new_loc) < 0.5:
            reward += -0.02
        
        for i in self.SensorObjList:
            dist = np.linalg.norm(self._agent_location - i.Location)
            if i.TotalContent != 0 and dist < 0.5:
                reward += 1
                i.TotalContent -= 0.001
                if i.TotalContent < 0:
                    self.MyDrone.Reward += 1
                    reward += 100
                    i.TotalContent = 0
        return reward

    def step(self, action):
        # Take a step in the environment with the given action and return the next observation, reward, terminated flag, and info dictionary
        action = np.clip(action, self.min_action, self.max_action)
        previous_action = self._agent_location.copy()
        self._agent_location = np.clip(self._agent_location + action, self.min_pos, self.max_pos)
        reward = self._reward_cal(previous_action, self._agent_location)
        
        terminated = False
        self.total_reward = self.total_reward + reward
        self.NumOfStep = self.NumOfStep + 1

        if (np.linalg.norm(self._agent_location - self._target_location) < 0.1 and self.NumOfStep > 1) or self.NumOfStep > 800:
            # reward = 10
            terminated = True
            self.last_episode = {"r": self.total_reward, "l": self.NumOfStep, "ReceivedSensor": self.MyDrone.Reward/self.SensorNum}

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info

    def render(self, mode='human'):
        # Render the environment
        pass
        
    def close(self):
        # Clean up any resources used by the environment
        pass
