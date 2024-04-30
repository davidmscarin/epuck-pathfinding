from controller import Robot, DistanceSensor
import numpy as np

robot: Robot = Robot()
timestep: int = int(robot.getBasicTimeStep())


def get_initial_coordinates():
    c = np.random.choice(["x","y"])
    if c == "x":
        x = np.random.choice([0.1,1.9])
        y = np.random.randint(1,19)/10
    else:
        y = np.random.choice([0.1,1.9])
        x = np.random.randint(1,19)/10
    return x,y

def getDistSensors():
    sensors = []
    for i in [0,1,2,3,4,5,6,7]:
        ds: DistanceSensor = robot.getDevice("ps" + str(i))
        ds.enable(timestep)
        sensors.append(ds)
    return sensors
def collision_detected(dist_sensors):
    THRESHOLD = 500
    if max(dist_sensors, key=lambda x: x.getValue()).getValue() > THRESHOLD:
        return True
    return False

def get_current_coordinates():
