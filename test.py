from controller import Robot
from bot_functions import collision_detected, reached_target, get_initial_coordinates, getDistSensors, getGPS, getLidar, getPointCloud, getTensor, euclidean_dist, manhattan_dist
from utils import cmd_vel
import torch
import torch.nn as nn
from torch.distributions import Normal


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


def load_model(name, input_dim = 16, output_dim = 2):

    model = PPO(input_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    epsilon = 0.2  # Clipping parameter for PPO
    gamma = 0.99  # Discount factor for rewards

    checkpoint = torch.load(name)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    episode = checkpoint['episode']
    loss = checkpoint['loss']

    return model, optimizer, epsilon, gamma, checkpoint, episode, loss


robot = Robot()
timestep = int(robot.getBasicTimeStep())
N_DIV = 8
lidar_sensors = getLidar(robot, timestep)
model, optimizer, epsilon, gamma, checkpoint, episode, loss = load_model('models/model_test2_run1000')

while robot.step(timestep) != -1:

    #get world state
    readingsX, readingsY = getPointCloud(lidar_sensors)
    state_tensor = torch.FloatTensor(getTensor(readingsX, readingsY, N_DIV))

    #get action
    action_distribution = model.forward(state_tensor)
    action = action_distribution.sample()

    #apply action
    lin_vel, ang_vel = action
    print(f"lin_vel: {lin_vel}\nang_vel: {ang_vel}")
    cmd_vel(robot, lin_vel, ang_vel)
