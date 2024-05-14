import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from controller import Robot, Supervisor, LidarPoint
from bot_functions import collision_detected, reached_target, get_initial_coordinates, getDistSensors, getGPS, getLidar, getPointCloud, getTensor, euclidean_dist
from utils import cmd_vel, warp_robot
import numpy as np

robot = Supervisor()
timestep = int(robot.getBasicTimeStep())
TARGET = [1.000, 1.000]
N_DIV = 8

class PPO(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PPO, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.sigma = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mu = self.fc3(x)
        sigma = torch.exp(self.sigma)
        return Normal(mu, sigma)

class Environment:
    def __init__(self, robot):
        self.robot = robot

    def reset(self):
        coords = get_initial_coordinates()
        warp_robot(robot, 'EPUCK', coords)
        pass

    def step(self, action):
        lin_vel, ang_vel = action
        cmd_vel(self.robot, lin_vel, ang_vel)
        robot.step()

def compute_rewards(dist_sensors, gps, TARGET):
    reward = 0
    if reached_target(gps, TARGET):
        reward += 10  # Reward for reaching the target
    if collision_detected(dist_sensors):
        reward -= 5  # Penalty for collision
    init_dist = euclidean_dist(gps, TARGET)
    
    final_dist = euclidean_dist(gps,TARGET)
    target_dist_gain = final_dist - init_dist
    # total_reward += ...

    return reward

def train():
    input_dim = N_DIV*2  # Example: number of LIDAR readings
    output_dim = 2  # Linear and angular velocities
    model = PPO(input_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    environment = Environment(robot)
    num_episodes = 1000
    max_timesteps = 200*timestep
    gps = getGPS(robot, timestep)
    dist_sensors = getDistSensors(robot, timestep)
    lidar_sensors = getLidar(robot, timestep)
    robot.step()

    for episode in range(num_episodes):
        environment.reset()
        total_reward = 0
        for t in range(max_timesteps):
            readingsX, readingsY = getPointCloud(lidar_sensors)
            state_tensor = torch.FloatTensor(getTensor(readingsX, readingsY, N_DIV))
            action_distribution = model.forward(state_tensor)
            action = action_distribution.sample()
            # print(action)
            environment.step(action.numpy())
            reward = compute_rewards(dist_sensors, gps, TARGET)
            total_reward += reward
            if collision_detected(dist_sensors) or reached_target(gps,target):
                break

        print(f"Episode {episode}, Total Reward: {total_reward}")

if __name__ == "__main__":
    env = Environment(robot)
    train()