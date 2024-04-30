from controller import Robot, DistanceSensor
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
    lidar = robot.getLidar('lidar')
    lidar.enable(timestep)
    lidar.enablePointCloud(timestep)
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
    readings = lidar.getPointCloud()
    return readings
