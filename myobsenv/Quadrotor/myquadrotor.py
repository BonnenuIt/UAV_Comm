import numpy as np

class MyQuadrotor:
    def __init__(self, Location, CommMode="Normal", Velocity=1, dim=3):
        self.Location = np.array(Location)  # 位置
        self.Reward = 0             # 收完的sensor数目
        self.InformationReward = 0  # 收的信息量总大小
        self.TotalTaskFinish = False# 是否完成全部任务并返回数据中心, 最后需要加个引导？？
        self.Velocity = Velocity    # 每秒走几个单位
        self.dim = dim
        self.HarvestOrNot = True    # 是否收集信息 (处理是否一直收集，还是只是悬停时收集)，没用
        self.HoverOrNot = False     # 是否悬停
        self.CommMode = CommMode    # 通信模式 "Hover" or "Normal"
        self.TotalTime = 0          # 用时，如果路径没走完，或者走完但是还在通信, 则计时

    def FuncHarvestOrNot(self, UAV_Sensor_Dist, CommRange, SensorsObj, map):
        '''
            UAV_Sensor_Dist : UAV - Sensor 距离
            CommRange : 可通信最大距离
            SensorsObj : Sensor 对象
            map : 提供地图的障碍物信息, 判断是否能够视距通信
        '''
        # 判断每个sensor是否这一秒通信
        if CommRange > UAV_Sensor_Dist \
        and SensorsObj.Finish == False \
        and not map.collision(self.Location, SensorsObj.Location) \
        and (self.CommMode == "Normal" or self.HoverOrNot == True):
        # 如果sensor在UAV通信范围内 且 
        # sensor信息没有传完 且 
        # sensor到UAV的视线通信路径没有被障碍物阻挡 且 
        # UAV 通信模式不是悬停通信，或者 UAV 正在悬停
            self.HarvestOrNot = True     # UAV要收集信息
        else: self.HarvestOrNot = False
