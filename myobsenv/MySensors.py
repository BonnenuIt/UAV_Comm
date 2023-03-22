import numpy as np

class MySensor:
    def __init__(self, index, Location, Content, Bandwidth, TxPower, NoisePower, CommAlpha, dim=3):
        self.index = index
        self.Location = np.array(Location)    # 位置
        self.TotalContent = Content      # 总内容
        self.Content = Content      # 还没有发的内容
        self.Finish = False         # 发完没有
        self.dim = dim

        # 通信配置，算香农容量
        self.Bandwidth = Bandwidth
        self.TxPower = TxPower
        self.NoisePower = NoisePower
        self.CommAlpha = CommAlpha

    def ContentComm(self, CommDist, Time, UAVObj, PrintNot = True): # sensor 信息通信
        CommSpeed = self.ShannonCapa(CommDist)
        # print(CommSpeed)
        # print(self.Location, '处的sensor的内容为', self.Content)
        if self.Content < 0:
            if PrintNot == True: print('Sensor', self.index, ': 内容为', self.Content, '  所有内容传输完成')
            self.Content = 0        # 信息内容为0
            self.Finish = True      # 完成传输
            UAVObj.Reward = UAVObj.Reward + 1   # 无人机获得收益
            return 1
        elif CommDist < 0 or Time < 0:  # error
            print("CommDist, Time 设置为负数")
            return -1
        
        UAVObj.InformationReward = UAVObj.InformationReward + CommSpeed * Time
        self.Content = self.Content - CommSpeed * Time  # 信息内容减少，通信容量*通信时间
        return 0
    
    def UAV_Sensor_Distance(self, UAVLoc):  # 计算与UAV距离
        return np.linalg.norm(self.Location - UAVLoc)
    
    def ShannonCapa(self, CommDist):
        return self.Bandwidth * np.log2(1 + (self.TxPower / self.NoisePower) * np.power(CommDist, -self.CommAlpha))
