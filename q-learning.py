from controller import Robot, Supervisor, LidarPoint
from bot_functions import collision_detected, reached_target, get_initial_coordinates, getDistSensors, getGPS, getLidar, getPointCloud, getTensor, euclidean_dist, turn
from utils import cmd_vel, warp_robot
import numpy as np
import random
import math

# Initialize the robot and sensors
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

# Define the target position (example coordinates)
TARGET = [1.000, 1.000]

# Define the action space
actions = {
    "left": (0, -math.pi/2),
    "right": (0, math.pi/2),
    "forward": (0.1, 0)
}

# Parameters for Q-learning
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate
num_episodes = 1000  # Number of training episodes
state_space_size = 100

# Initialize Q-table (assuming state space of size 100)
q_table = np.zeros((state_space_size, len(actions)))


class Environment:
    def __init__(self, robot):
        self.robot = robot
        self.gps = getGPS(robot, timestep)
        self.dist_sensors = getDistSensors(robot, timestep)
        self.lidar_sensors = getLidar(robot, timestep)

    def reset(self):
        coords = get_initial_coordinates()
        warp_robot(robot, 'EPUCK', coords)
        return self.get_state()

    def step(self, action):
        lin_vel, ang_vel = action
        cmd_vel(self.robot, lin_vel, ang_vel)
        self.robot.step(int(self.robot.getBasicTimeStep()))
        next_state = self.get_state()
        reward = self.compute_rewards()
        done = reward == 10 or self.collision_detected()
        return next_state, reward, done

    def get_state(self):
        position = self.gps.getValues()
        #dist_values = [sensor.getValue() for sensor in self.dist_sensors]
        # Combine position and distance sensor readings to form the state
        state = position
        print(np.sum(np.array(state)))
        return np.sum(np.array(state))

    def compute_rewards(self):
        reward = 0
        if self.reached_target():
            reward += 10  # Reward for reaching the target
        if self.collision_detected():
            reward -= 5  # Penalty for collision
        return reward

    def reached_target(self):
        position = self.gps.getValues()
        return math.sqrt((position[0] - TARGET[0]) ** 2 + (position[2] - TARGET[1]) ** 2) < 0.1

    def collision_detected(self):
        for sensor in self.dist_sensors:
            if sensor.getValue() < 1000:  # Adjust threshold based on your setup
                return True
        return False

# Define action functions
def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.choice(list(actions.keys()))  # Explore
    else:
        return list(actions.keys())[np.argmax(q_table[state])]  # Exploit

def update_q_table(state, action, reward, next_state):
    best_next_action = np.argmax(q_table[next_state])
    td_target = reward + gamma * q_table[next_state, best_next_action]
    td_error = td_target - q_table[state, action]
    q_table[state, action] += alpha * td_error


def train():
    # Training the robot
    robot.step()
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action_key = choose_action(state)
            action = actions[action_key]
            next_state, reward, done = env.step(action)

            update_q_table(state, list(actions.keys()).index(action_key), reward, next_state)

            state = next_state

if __name__ == '__main__':
    env = Environment(robot)
    train()
    print("Training done")
