from controller import Robot, DistanceSensor
from utils import cmd_vel
import time

# Create the Robot instance.
robot: Robot = Robot()

timestep: int = int(robot.getBasicTimeStep())  # in ms

# TODO
def getDistSensors():
    sensors = []
    for i in [0,1,2,3,4,5,6,7]:
        ds: DistanceSensor = robot.getDevice("ps" + str(i))
        ds.enable(timestep)
        sensors.append(ds)
    return sensors

def obstacle_detected(dist_sensors):
    THRESHOLD = 500
    if max(dist_sensors, key=lambda x: x.getValue()).getValue() > THRESHOLD:
        return True
    return False

def controller():
    dist_sensors: [DistanceSensor] = getDistSensors()
    i=0
    while True:
        cmd_vel(robot, 0.12, 0)
        while not obstacle_detected(dist_sensors):
            readings = []
            for ds in dist_sensors:
                readings.append(ds.getValue())
            robot.step()
            # time.sleep(1)
            print(readings)
        cmd_vel(robot, 0, 0.15)
        while obstacle_detected(dist_sensors):
            readings = []
            for ds in dist_sensors:
                readings.append(ds.getValue())
            robot.step()
            # time.sleep(1)
            print(readings)


if __name__ == "__main__":
    controller()