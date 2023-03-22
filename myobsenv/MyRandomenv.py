import math
import numpy as np
from .MyLine import SensorObstaclesJudge, SensorHight, Line2Points
import copy
from .MySensors import MySensor
from .Quadrotor.myquadrotor import MyQuadrotor
from .PathPlanning import mymaputils

class MyRandomenv():
    def __init__(self):
        self.MyVelocity = 1

        # self.size = size
        self.bounds = np.array([0,100])
        self.StepTime = 1                   # 图中1个点表示1s
        self.start = np.array([0, 0, 50])    # 数据中心位置
        self.UAVLoc = self.start
        self.dim = 3
        
        self.action_MaxDistance = 10        # 一个episode最大的动作距离，即一次动作耗费的恒定时间，变速
        self.max_action = [self.action_MaxDistance, math.pi, 2*math.pi]
        self.CommRange = 20                 # 最大通信距离
        self.CommMode = "Normal"            # UAV 通信模式 "Hover" or "Normal"，悬停通信还是飞行过程也可以通信
        self.Bandwidth = 100
        self.TxPower = 1
        self.NoisePower = 1
        self.CommAlpha = 2
        print("10m通信容量为", self.Bandwidth * np.log2(1 + (self.TxPower / self.NoisePower) * np.power(10.0, -self.CommAlpha)), '单位每秒')

        # self.ObsIterNum = 13
        # self.SensorNum = 10
        self.StateObsMap = None
        self.StateSensorMap = None
        self.StateSensorContentMap = None
        self.UAVLocMap = None
        # self.reset()

    def ObsGen(self, MaxNumObs, Low, High):
        obstacle_res = []
        for _ in range(MaxNumObs):
            ObsValid = True
            lx = np.random.randint(Low[0], High[0] * 0.8)
            ly = np.random.randint(Low[1], High[1] * 0.8)
            hx = np.random.randint(lx+1, High[0])
            hy = np.random.randint(ly+1, High[1])
            
            # https://blog.csdn.net/qianchenglenger/article/details/50484053
            for j in obstacle_res:
                if not (hx < j[0] or hy < j[1] or lx > j[3] or ly > j[4]):  # 生成的与之前的重叠
                    ObsValid = False
                    break
            if ObsValid == False:   # 生成的与之前的重叠
                continue
            hz = np.random.randint(Low[2], High[2])
            obstacle_res.append([lx, ly, 0, hx, hy, hz])    
            # 3D boxes   lx, ly, lz, hx, hy, hz
        return obstacle_res

    def EnvMatrix(self):
        self.StateObsMap = np.zeros((self.bounds[1], self.bounds[1]))       # 2D 图，3D障碍物图压扁，元素值为高度/10
        for Obs_i in self.obstacles: # 3D boxes   lx, ly, lz, hx, hy, hz
            for x in range(Obs_i[0], Obs_i[3] + 1):
                for y in range(Obs_i[1], Obs_i[4] + 1):
                    self.StateObsMap[x, y] = Obs_i[5] / 10
        self.StateSensorMap = np.zeros((self.bounds[1], self.bounds[1]))    # 2D 图，3D Sensor图压扁，元素值为高度
        for Sensor_i in self.SensorObjList: # x,y,z
            self.StateSensorMap[Sensor_i.Location[0], Sensor_i.Location[1]] = Sensor_i.Location[2] / 10
    
    def StateMatrix(self):
        '''
        Sensor Content / UACLoc is changing, so an independent function to build the content map and change the state
        '''
        self.StateSensorContentMap = np.zeros((self.bounds[1], self.bounds[1]))        # 2D 图，3D Sensor图压扁，元素值为Sensor信息量
        for i in range(len(self.SensorObjList)): # x,y,content
            Sensor_i = self.SensorObjList[i]
            self.StateSensorContentMap[Sensor_i.Location[0], Sensor_i.Location[1]] = Sensor_i.Content
        self.UAVLocMap = np.zeros((self.bounds[1], self.bounds[1]))         # 2D , UAV 坐标
        # for i in range(self.UAVLoc.shape[0]):
        #     self.UAVLocMap[34*i : 34*(i+1), :] = self.UAVLoc[i]
        UAVLocTemp = np.clip(np.array([int(self.UAVLoc[0]), int(self.UAVLoc[1])]), self.bounds[0], self.bounds[1] - 1)
        self.UAVLocMap[UAVLocTemp[0], UAVLocTemp[1]] = self.UAVLoc[2]

        statei = np.concatenate((self.StateObsMap[np.newaxis, :, :], 
                                self.StateSensorMap[np.newaxis, :, :], 
                                self.StateSensorContentMap[np.newaxis, :, :], 
                                self.UAVLocMap[np.newaxis, :, :]), axis=0)
        return statei
    
    def StateMatrix1D(self):
        OBS = np.array(self.obstacles).flatten() / 10
        MaxObsDim = self.ObsIterNum * len(self.obstacles[1])
        OBS = np.pad(OBS, (0, MaxObsDim - OBS.shape[0]),'constant', constant_values=(0,0))
        SEN = []
        for i in range(len(self.SensorObjList)): # x,y,z
            Sensor_i = self.SensorObjList[i]
            Sensor_i_loc = (Sensor_i.Location / 10).tolist()
            Sensor_i_loc.append(Sensor_i.Content)
            SEN.append(Sensor_i_loc)
        MaxObsDim = self.ObsIterNum * len(self.obstacles[1])
        SEN = np.array(SEN).flatten()
        SEN = np.pad(SEN, (0, self.SensorNum * 4 - SEN.shape[0]), 'constant', constant_values=(0,0))
        res = np.concatenate((OBS, SEN, self.UAVLoc/10), axis=0)
        return res

    def action_RandomSample(self):
        r = np.random.rand() * self.max_action[0]
        theta = np.random.rand() * self.max_action[1]
        phi = np.random.rand() * self.max_action[2]
        self.actioni = np.array([r, theta, phi])
        return self.actioni

    def reset(self, StateDim=2):
        '''
        reset a new random environment 重置地图, random obstacles and random sensors
        return the state
        '''
        self.UAVLoc = self.start
        self.obstacles = []
        self.ObsGen()
        
        self.SensorNodes = (np.random.rand(self.SensorNum, 2) * self.bounds[1]).astype(int).tolist()
        self.SensorContent = (np.random.rand(self.SensorNum) * 10).tolist()   # sensor 信息内容大小
        self.SensorNodes = SensorHight(self.SensorNodes, self.obstacles, HightL=1, HightH=20)      # sensor如果初步设定到障碍物内，则放到障碍物顶上
        
        self.SensorObjList = []
        for i in range(len(self.SensorNodes)):
            self.SensorObjList.append(MySensor(index=i+1, Location=self.SensorNodes[i], 
                                               Content=self.SensorContent[i], 
                                               Bandwidth=self.Bandwidth, TxPower=self.TxPower, 
                                               NoisePower=self.NoisePower, CommAlpha=self.CommAlpha))
        self.MyDrone = MyQuadrotor(Location=self.start, CommMode=self.CommMode, Velocity=None)

        if StateDim == 2:
            self.EnvMatrix()
            self.state = self.StateMatrix()
        elif StateDim == 1:
            self.state = self.StateMatrix1D()
        self.StateMatrix1D()

        self.CummulatedCommRewardOld = 0   # 所有之前的通信reward相加，包含时间的惩罚
        
        return self.state
    
    def step(self, action, mapobs, StateDim=2):
        '''
        new Rectangular position of UAV needs to be generated by Spherical coordinate
        '''
        # newUAVPos_Spherical = self.cartesian_to_spherical(self.UAVLoc) + action # action 为球坐标系
        action_cartesian = self.spherical_to_cartesian(action)
        # self.UAVLocOld = copy.deepcopy(self.UAVLoc)
        # self.UAVLocNew = self.spherical_to_cartesian(newUAVPos_Spherical)  # UAVLoc 为直角坐标系
        self.UAVLocNew = self.UAVLoc + action
        self.UAVLocNew = np.clip(self.UAVLocNew, self.bounds[0], self.bounds[1])
        PathPoints = Line2Points(Start=self.UAVLoc, End=self.UAVLocNew, PointNum=self.action_MaxDistance)   # 该动作途径的所有点，最后一个点不作通信计算
        for i in range(self.action_MaxDistance):    # 循环 PathPoints 点数 (一个动作), 包括无人机此时的起点
            self.MyDrone.Location = PathPoints[i, :]
            self.MyDrone.TotalTime = self.MyDrone.TotalTime + 1
            self.MyDrone.HoverOrNot = False

            for SensorsObj in self.SensorObjList:   # 和每个sensor通信
                UAV_Sensor_Dist = SensorsObj.UAV_Sensor_Distance(self.MyDrone.Location)
                self.MyDrone.FuncHarvestOrNot(UAV_Sensor_Dist, self.CommRange, SensorsObj, mapobs)
                # 判断每个sensor是否这一秒通信, 如果sensor在UAV通信范围内 且 sensor信息没有传完 且 sensor到UAV的视线通信路径没有被障碍物阻挡 且 UAV 通信模式不是悬停通信，或者 UAV 正在悬停
                if self.MyDrone.HarvestOrNot == True:     # UAV要收集信息
                    SensorsObj.ContentComm(CommDist=UAV_Sensor_Dist, Time=1, UAVObj=self.MyDrone, PrintNot=False) # 一个点是一秒，time=1，通信
        # self.MyDrone.Location = self.UAVLocNew
        UAVLOCTemp = copy.deepcopy(self.UAVLoc)
        self.UAVLoc = copy.deepcopy(self.UAVLocNew)
        if StateDim == 2:
            self.state = self.StateMatrix() # 此处是next state
        elif StateDim == 1:
            self.state = self.StateMatrix1D()
        # self.StateMatrix1D()

        if self.MyDrone.Reward == len(self.SensorObjList):
            self.MyDrone.TotalTaskFinish == True
        AllReward = self.reward_cal(self.MyDrone.Reward, self.MyDrone.InformationReward, self.MyDrone.TotalTime) - self.CummulatedCommRewardOld
        # 计算这一个 episode 内, 单独的reward
        self.CummulatedCommRewardOld = self.reward_cal(self.MyDrone.Reward, self.MyDrone.InformationReward, self.MyDrone.TotalTime)

        # AllReward = AllReward + 2   #
        if np.linalg.norm(UAVLOCTemp - self.UAVLoc) > 3:
            AllReward = 0.2
        else:
            AllReward = 0
        return self.state, AllReward, self.MyDrone.TotalTaskFinish
    
    def reward_cal(self, NumReward, InforReward, TimeReward):
        return NumReward * 5 + InforReward - TimeReward * 0

    def cartesian_to_spherical(self, UAVLoc):  # 输出为弧度
        x = UAVLoc[0] + 1e-7
        y = UAVLoc[1]
        z = UAVLoc[2]
        r = math.sqrt((x**2) + (y**2) + (z**2)) + 1e-7
        theta = math.acos(z / r)
        phi = math.atan(y / x)
        return np.array([r, theta, phi])

    def spherical_to_cartesian(self, UAVLoc_Spherical):    # 输入为弧度
        r = UAVLoc_Spherical[0]
        theta = UAVLoc_Spherical[1]
        phi = UAVLoc_Spherical[2]
        x = r * (math.sin(theta)) * (math.cos(phi))
        y = r * (math.sin(theta)) * (math.sin(phi))
        z = r * (math.cos(theta))
        # if x > self.bounds[1]: x = self.bounds[1]
        # elif x < self.bounds[0]: x = self.bounds[0]
        # if y > self.bounds[1]: y = self.bounds[1]
        # elif y < self.bounds[0]: y = self.bounds[0]
        # if z > self.bounds[1]: z = self.bounds[1]
        # elif z < self.bounds[0]: z = self.bounds[0]
        return np.array([x, y, z])


def evaluate_policy(env, agent):
    times = 3  # Perform three evaluations and calculate the average
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset()
        mapobs = mymaputils.myMap(env.obstacles, env.SensorNodes, env.bounds, dim = env.dim)
        done = False
        episode_reward = 0
        for episode_steps in range(max_episode_steps):
            action = agent.choose_action(s)  # We do not add noise when evaluating
            s_, reward, done = env.step(action=action, mapobs=mapobs)
            if done: # 限定episode之前，完成所有传感器收集, 完成点到终点的距离也是惩罚
                reward = reward + 1000 - np.linalg.norm(env.start - env.UAVLoc)
            
            episode_reward += reward
            s = copy.deepcopy(s_)
            if done:
                break
        evaluate_reward += episode_reward

    return int(evaluate_reward / times)

def sarsPlot(s, action, reward, s_, env):
    print('#####', (s[0:1]==s_[0:1]).all(), end=' ')
    loc1 = np.array([s[3, 0, 0], s[3, 50, 0], s[3, 99, 0]]) # s 中的UAV坐标，最后一层
    # loc2 = env.cartesian_to_spherical(loc1) + action # action 为球坐标系
    action_cartesian = env.spherical_to_cartesian(action)
    # loc2 = env.spherical_to_cartesian(loc2)  # UAVLoc 为直角坐标系
    loc2 = loc1 + action_cartesian
    loctrue = np.array([s_[3, 0, 0], s_[3, 50, 0], s_[3, 99, 0]])
    print(loc2, (loc2 == loctrue).all(), end=' ')
    print((s[2]==s_[2]).all(), end=' ')
    print(np.sum(s[2] - s_[2]), end=' ')
    print(reward, end=' ')
    print(reward - np.sum(s[2] - s_[2]), '#####')
    # print()

if __name__ == '__main__':
    '''
    module load anaconda3/py38 cudnn8.0-cuda11.1 cuda11.1
    srun --gres=gpu:1 -w node04 python MyRandomenv.py'''
    from MyDDPG import ReplayBuffer, DDPG
    from torch.utils.tensorboard import SummaryWriter
    # TODO: 撞障碍物时的惩罚没加, 时间的删除了, UAVloc转矩阵的需要换成稀疏的相对位置，往传感器的方向设置reward

    max_train_steps = 40  # Maximum number of training steps
    max_episode_steps = 200  # Maximum number of episode
    update_freq = 300
    evaluate_freq = 400  # Evaluate the policy every 'evaluate_freq' steps
    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    randomActionProb = 0.95
    randomActionProbMin = 0.1
    env = MyRandomenv()
    env_evaluate = MyRandomenv()
    agent = DDPG(action_dim=3, max_action=env.max_action)
    replay_buffer = ReplayBuffer(state_dim=[4, env.bounds[1], env.bounds[1]], action_dim=3, max_size=4000)
    log_dir = 'runs/DDPG_{}'.format(1)
    writer = SummaryWriter(log_dir)

    for epoch in range(max_train_steps): # Record the total steps during the training
        s = env.reset()
        mapobs = mymaputils.myMap(env.obstacles, env.SensorNodes, env.bounds, dim = env.dim)
        done = False
        # print("new epoch")
        for episode_steps in range(max_episode_steps): # 记录 episode, 即 Quadrotor.TotalTime
            randomActionProb = max(randomActionProb * 0.999, randomActionProbMin)
            if np.random.rand() < randomActionProb:
                action = env.action_RandomSample()
            else:
                action = agent.choose_action(s)
            
            s_, reward, done = env.step(action=action, mapobs=mapobs)

            if done: # 限定episode之前，完成所有传感器收集, 完成点到终点的距离也是惩罚
                reward = reward + 1000 - np.linalg.norm(env.start - env.UAVLoc)
            
            replay_buffer.store(s, action, reward, s_, done)
            # sarsPlot(s, action, reward, s_, env)

            s = copy.deepcopy(s_)

            total_steps = (epoch * max_episode_steps) + episode_steps + 1
            # print(epoch, total_steps, end = ' ')
            print("#", end='')
            # Take 50 steps,then update the networks 50 times
            if total_steps % update_freq == 0:
                for _ in range(update_freq):
                    agent.learn(replay_buffer)

            # Evaluate the policy every 'evaluate_freq' steps
            if (total_steps + 1) % evaluate_freq == 0:
                evaluate_num += 1
                evaluate_reward = evaluate_policy(env_evaluate, agent)
                evaluate_rewards.append(evaluate_reward)
                print("\nevaluate_num:{} \t evaluate_reward:{}".format(evaluate_num, evaluate_reward))
                writer.add_scalar('step_rewards', evaluate_reward, global_step=total_steps)
                # Save the rewards
                if evaluate_num % 10 == 0:
                    np.save(log_dir + '.npy', np.array(evaluate_rewards))

            # print(int(reward), end='  ')
            if done:
                break
        
    print()
