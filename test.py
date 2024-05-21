from controller import Robot, Lidar
from utils import cmd_vel
from bot_functions import turn
import math
import time
robot = Robot()

timestep = int(robot.getBasicTimeStep())
dir = "right"

while robot.step() != -1:
    turn(robot, dir)
    time.sleep(3)
