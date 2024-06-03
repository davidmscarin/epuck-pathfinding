import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from controller import Robot, Supervisor, LidarPoint
from bot_functions import collision_detected, reached_target, get_initial_coordinates, getDistSensors, getGPS, getLidar, getPointCloud, getTensor, euclidean_dist, manhattan_dist
from utils import cmd_vel, warp_robot
import numpy as np
import time

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

def compute_dist(system, init_dist, gps, TARGET):
    if system == 'Euclidean':
        final_dist = euclidean_dist(gps, TARGET)
        #target_dist_gain = init_dist - final_dist  # Calculating the distance gain
        #print("dist gain: ", dist_gain)
        return final_dist

    elif system == 'Manhattan':
        final_dist = manhattan_dist(gps, TARGET)
        return final_dist

def compute_rewards(system, TARGET, gps, dist_sensors, init_dist = 0, get_dist = True):
    reward = 0
    # print(get_dist)
    if reached_target(gps, TARGET):
        reward = 3  # Reward for reaching the target
    elif collision_detected(dist_sensors):
        reward = -3  # Penalty for collision
    elif get_dist:
        dist = -compute_dist(system, init_dist, gps, TARGET)
        reward = round(dist,8)

    # print("Reward: ", reward)
    return reward

def compute_ppo_loss(log_probs, advantages, epsilon):
    # Convert advantages to tensor
    advantages_tensor = torch.FloatTensor(advantages)

    # Compute old policy probabilities
    old_probs = torch.exp(log_probs)

    # Compute ratio of new policy probabilities to old policy probabilities
    new_probs = torch.exp(log_probs)
    ratio = new_probs / old_probs

    # Compute surrogate loss
    clip_range = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    surrogate_loss = torch.min(ratio * advantages_tensor, clip_range * advantages_tensor)
    surrogate_loss = -torch.mean(surrogate_loss)  # Negative because we want to maximize the objective

    return surrogate_loss

def discount_rewards(rewards, gamma):
    discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(rewards))):
        running_add = running_add * gamma + rewards[t]
        discounted_rewards[t] = running_add
    return discounted_rewards


def train(load = False, model_name = None):

    input_dim = N_DIV * 2  # Example: number of LIDAR readings
    output_dim = 2  # Linear and angular velocities
    model = PPO(input_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    epsilon = 0.2  # Clipping parameter for PPO
    gamma = 0.99  # Discount factor for rewards

    if load:
        checkpoint = torch.load(model_name)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        episode = checkpoint['episode']
        loss = checkpoint['loss']

    loss_over_time = []  # Stores the loss after each episode
    reward_over_time = []  # Stores the reward after each episode

    #env variables
    environment = Environment(robot)
    num_episodes = 1000
    max_timesteps = 500
    save_rate = 100

    #robot sensors
    gps = getGPS(robot, timestep)
    dist_sensors = getDistSensors(robot, timestep)
    lidar_sensors = getLidar(robot, timestep)
    robot.step()

    #training loop
    print(f"Running {num_episodes} episodes")
    for episode in range(num_episodes):
        print(f"Episode {episode+1} started")

        #other variables
        i_t = time.time()
        environment.reset()
        init_dist = euclidean_dist(gps, TARGET)
        total_reward = 0
        log_probs = []  # Store log probabilities of actions taken
        rewards = []  # Store rewards obtained

        print(f"Running {max_timesteps} timesteps")
        for t in range(max_timesteps):
            readingsX, readingsY = getPointCloud(lidar_sensors)
            state_tensor = torch.FloatTensor(getTensor(readingsX, readingsY, N_DIV))
            action_distribution = model.forward(state_tensor)
            action = action_distribution.sample()

            environment.step(action.numpy())

            step_reward = compute_rewards("Manhattan", TARGET, gps, dist_sensors, init_dist)
            total_reward += step_reward

            log_prob = action_distribution.log_prob(action)
            log_probs.append(log_prob)
            rewards.append(step_reward)

            if step_reward <= -3 or step_reward >= 3:
                print(f"Stopped early. Timestep: {t}")
                break

        print(rewards)
        reward_over_time.append(total_reward)

        # Convert rewards to numpy array and compute advantages
        rewards = np.array(rewards)
        rewards = rewards.repeat(2, 0)
        # print(rewards)
        advantages = discount_rewards(rewards, gamma)

        # Convert log_probs to tensor
        log_probs_tensor = torch.cat(log_probs)
        print(f"Advantages:{advantages.shape}")
        print(f"log probabilities:{log_probs_tensor.shape}")
        print(f"Epsilon:{epsilon}")

        # Compute surrogate loss
        loss = compute_ppo_loss(log_probs_tensor, advantages, epsilon)
        print(f"Loss: {loss}")
        loss_over_time.append(loss)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"Episode {episode} Finished\nTotal Reward: {total_reward}")
        print()

        # save last episode
        if (episode + 1) % save_rate == 0 or episode == 0:
            print(episode + 1)
            print(save_rate)
            torch.save({
                'episode': episode,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, 'models/model_test2_run'+str(episode+1))

    np.save("loss_over_time", np.array(loss_over_time))
    np.save("reward_over_time", np.array(reward_over_time))


if __name__ == "__main__":
    env = Environment(robot)
    train()