from controller import Robot, Lidar
from utils import cmd_vel
robot = Robot()

timestep = int(robot.getBasicTimeStep())

lidar_sensors = Lidar('lidar')
lidar_sensors.enable(timestep)
lidar_sensors.enablePointCloud()

while robot.step() != -1:
    #cmd_vel(robot, 0.1, 0)
    point_cloud = lidar_sensors.getPointCloud()
    coords = []
    for point in point_cloud:
        coords.append(point.x)
        coords.append(point.y)
    print(coords)
    print(len(coords))