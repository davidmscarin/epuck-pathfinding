from controller import Robot, DistanceSensor, Lidar
import numpy as np
import math
from utils import cmd_vel

def get_initial_coordinates():
    c = np.random.choice(["x","y"])
    if c == "x":
        x = np.random.choice([0.1,1.9])
        y = np.random.randint(1,19)/10
    else:
        y = np.random.choice([0.1,1.9])
        x = np.random.randint(1,19)/10
    return (x,y)

def getDistSensors(robot, timestep):
    sensors = []
    for i in range(8):
        ds: DistanceSensor = robot.getDevice("ps" + str(i))
        ds.enable(timestep)
        sensors.append(ds)
    return sensors

def getGPS(robot, timestep):
    gps = robot.getDevice('gps')
    gps.enable(timestep)
    return gps

def getLidar(robot, timestep):
    lidar = Lidar('lidar')
    lidar.enable(timestep)
    lidar.enablePointCloud()
    return lidar

def collision_detected(dist_sensors):
    THRESHOLD = 500
    if max(dist_sensors, key=lambda x: x.getValue()).getValue() > THRESHOLD:
        return True
    return False

def reached_target(gps, target):
    readings = gps.getValues()
    if round(readings[0], 3) == target[0] and round(readings[1], 3) == target[1]:
        return True
    return False

def getPointCloud(lidar):
    point_cloud = lidar.getPointCloud()
    readingsX = []
    readingsY = []
    for point in point_cloud:
        x = point.x
        y = point.y
        readingsX.append(x)
        readingsY.append(y)
    return readingsX[:-1], readingsY[:-1]

def getTensor(readingsX, readingsY, n_div):
    tensor = []
    for j in range(n_div):
        sumX = 0
        sumY = 0
        for i in range(len(readingsX)//n_div):
            if not math.isinf(readingsX[j*i]):
                sumX += readingsX[j*i]
            if not math.isinf(readingsX[j*i]):
                sumY += readingsY[j * i]
        tensor.append(sumX/n_div)
        tensor.append(sumY/n_div)
    return tensor

def euclidean_dist(gps, target):
    readings = gps.getValues()
    x, y = readings[0], readings[1]
    return math.sqrt((x-target[0])**2 + (y-target[1])**2)


def turn(robot, direction, timestep=500):

    if direction == "right":
        cmd_vel(robot, 0, math.pi/2)
        robot.step(timestep)

    elif direction == "left":
        cmd_vel(robot, 0, -math.pi/2)
        robot.step(timestep)

    elif direction == 'forward':
        cmd_vel(robot, linear_vel=0.1, angular_vel=0)
        robot.step(timestep)