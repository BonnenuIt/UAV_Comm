import numpy as np

def Line2Points(Start, End, PointNum):  # 得到Start, End两点之间的所有点，点数为 PointNum
    Points = np.array([Start + x * (End - Start) / PointNum for x in range(PointNum)])
    return Points

def SensorObstaclesJudge(Sensor, Obstacle):
    for i in range(len(Sensor)):
        for Obstacle_j in Obstacle:
            if Sensor[i][0] >= Obstacle_j[0] and Sensor[i][0] <= Obstacle_j[3]:         # x轴在障碍物内
                if Sensor[i][1] >= Obstacle_j[1] and Sensor[i][1] <= Obstacle_j[4]:     # y轴在障碍物内
                    if Sensor[i][2] >= Obstacle_j[2] and Sensor[i][2] <= Obstacle_j[5]: # z轴在障碍物内
                        Sensor[i][2] = Obstacle_j[5] + 1                                # sensor放到障碍物顶上，不加1会被视为障碍物一部分
                        break
    return Sensor

def SensorHight(Sensor, Obstacle, HightL=0, HightH=1):
    for i in range(len(Sensor)):
        Sensor[i].append(int((np.random.rand() * (HightH - HightL)) + HightL))       # 随机生成 sensor 的高度
        for Obstacle_j in Obstacle:
            if Sensor[i][0] >= Obstacle_j[0] and Sensor[i][0] <= Obstacle_j[3]:         # x轴在障碍物内
                if Sensor[i][1] >= Obstacle_j[1] and Sensor[i][1] <= Obstacle_j[4]:     # y轴在障碍物内
                    if Sensor[i][2] >= Obstacle_j[2] and Sensor[i][2] <= Obstacle_j[5]: # z轴在障碍物内
                        Sensor[i][2] = Obstacle_j[5] + 1                                # sensor放到障碍物顶上，不加1会被视为障碍物一部分
                        break
    return Sensor

print("Import MyLib")

if __name__ == '__main__':
    Start = np.array([0,0,0])
    End = np.array([-1,-1.5,2])
    Points = Line2Points(Start=Start, End=End, PointNum=100)
    print(Points)

    print(bool(0))

    def update_plot(self, UAVObj, ax):
        UAVLoc = UAVObj.Location
        self.pos_history.append(UAVLoc)
        history = np.array(self.pos_history)
        # print('shape = ', history.shape, '   shape1 = ', history[:,0].shape, '   shape2 = ', history[:,1].shape)
        self.lines[-1].set_data(history[:,0], history[:,1]) # 路径绿点
        self.lines[-1].set_3d_properties(history[:,-1])     # 路径绿点的高 3D
        SensorLines = []
        for SensorsObj in self.SensorObjList:
            UAV_Sensor_Dist = SensorsObj.UAV_Sensor_Distance(UAVLoc)
            if self.CommRange > UAV_Sensor_Dist \
            and SensorsObj.Finish == False \
            and not self.map.collision(UAVLoc, SensorsObj.Location) \
            and (UAVObj.CommMode == "Normal" or UAVObj.HoverOrNot == True):
            # 如果sensor在UAV通信范围内 且 
            # sensor信息没有传完 且 
            # sensor到UAV的视线通信路径没有被障碍物阻挡 且 
            # UAV 通信模式不是悬停通信，或者 UAV 正在悬停
                UAVObj.HarvestOrNot = True     # UAV要收集信息
            else: UAVObj.HarvestOrNot = False
                
            if UAVObj.HarvestOrNot == True:     # UAV要收集信息
                SensorsObj.ContentComm(CommDist=UAV_Sensor_Dist, Time=1, UAVObj=UAVObj)
                SensorLoc = SensorsObj.Location
                ln = ax.plot3D([SensorLoc[0], UAVLoc[0]], \
                    [SensorLoc[1], UAVLoc[1]], \
                    [SensorLoc[2], UAVLoc[2]])  # UAV和sensor的连接线
                SensorLines.append(ln)      # 保存连接线，为了后面删除
        return SensorLines


    def update_plot(self, UAVObj, ax):
        UAVLoc = UAVObj.Location
        self.pos_history.append(UAVLoc)
        history = np.array(self.pos_history)
        # print('shape = ', history.shape, '   shape1 = ', history[:,0].shape, '   shape2 = ', history[:,1].shape)
        self.lines[-1].set_data(history[:,0], history[:,1]) # 路径绿点
        self.lines[-1].set_3d_properties(history[:,-1])     # 路径绿点的高 3D
        SensorLines = []
        for SensorsObj in self.SensorObjList:
            UAV_Sensor_Dist = SensorsObj.UAV_Sensor_Distance(UAVLoc)
            if UAVObj.HarvestOrNot == True and self.CommRange > UAV_Sensor_Dist:
                if SensorsObj.Finish == False and not self.map.collision(UAVLoc, SensorsObj.Location):
                # UAV要收集信息 且 如果sensor在UAV通信范围内 且 sensor信息没有传完 且 sensor到UAV的视线通信路径没有被障碍物阻挡
                    if UAVObj.CommMode == "Normal" or UAVObj.HoverOrNot == True:
                    # UAV 通信模式不是悬停通信，或者 UAV 正在悬停
                        SensorsObj.ContentComm(CommDist=UAV_Sensor_Dist, Time=1, UAVObj=UAVObj)
                        SensorLoc = SensorsObj.Location
                        ln = ax.plot3D([SensorLoc[0], UAVLoc[0]], [SensorLoc[1], UAVLoc[1]], [SensorLoc[2], UAVLoc[2]])  # UAV和sensor的连接线
                        SensorLines.append(ln)      # 保存连接线，为了后面删除
        return SensorLines
    