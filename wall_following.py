from controller import Robot, DistanceSensor
from utils import cmd_vel

# Create the Robot instance.
robot: Robot = Robot()

timestep: int = int(robot.getBasicTimeStep())  # in ms

# TODO
def getDistSensors():
    sensors = []
    for i in [0,1,6,7]:
        ds: DistanceSensor = robot.getDevice("ps" + str(i))
        ds.enable(timestep)
        sensors.append(ds)
    return sensors

def obstacle_detected(dist_sensors):
    THRESHOLD = 120
    for ds in dist_sensors:
        if ds.getValue() > THRESHOLD:
            return True
    return False

def controller():
    dist_sensors: [DistanceSensor] = getDistSensors()
    i=0
    while True:
        readings = []
        for ds in dist_sensors:
            readings.append(ds.getValue())
        print(readings)
        cmd_vel(robot, 0.12, 0)
        while not obstacle_detected(dist_sensors):
            robot.step()
        cmd_vel(robot, 0, 0.15)
        while obstacle_detected(dist_sensors):
            robot.step()


if __name__ == "__main__":
    controller()