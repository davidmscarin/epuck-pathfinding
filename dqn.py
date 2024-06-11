import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from bot_functions import take_action, collision_detected, reached_target, get_initial_coordinates, getDistSensors, getGPS, getLidar, getPointCloud, getTensor, euclidean_dist, manhattan_dist
from controller import Supervisor
from utils import cmd_vel, warp_robot
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")


robot = Supervisor()
timestep = int(robot.getBasicTimeStep())
TARGET = [1.000, 1.000]
N_DIV = 8

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class Environment:
    def __init__(self, robot, TARGET, gps, dist_sensors, lidar_sensors):
        self.robot = robot
        self.TARGET = TARGET
        # initialize robot sensors
        self.gps = gps
        self.dist_sensors = dist_sensors
        self.lidar_sensors = lidar_sensors
        self.action_space = [0, 1, 2]
        self.action_dict = {0: "forward", 1: "right", 2: "left"}

    def compute_dist(self, system, gps, TARGET):
        if system == 'Euclidean':
            final_dist = euclidean_dist(gps, TARGET)
            # target_dist_gain = init_dist - final_dist  # Calculating the distance gain
            # print("dist gain: ", dist_gain)
            return final_dist

        elif system == 'Manhattan':
            final_dist = manhattan_dist(gps, TARGET)
            return final_dist

    def compute_rewards(self, system = 'Euclidean'):
        reward = 0

        if reached_target(self.gps, self.TARGET):
            reward = 3  # Reward for reaching the target
            return reward
        elif collision_detected(self.dist_sensors):
            reward = -3  # Penalty for collision
            return reward

        dist = -self.compute_dist(system, self.gps, self.TARGET)
        reward = round(dist, 8)

        return reward

    def reset(self):
        coords = get_initial_coordinates()
        warp_robot(robot, 'EPUCK', coords)
        pass
    def step(self, action):

        truncated = False
        terminated = False

        reward = self.compute_rewards()

        if reward >= 3:
            terminated = True
        elif reward < -3:
            truncated = True

        action = self.action_dict[action]
        take_action(robot, action)

        return reward, truncated, terminated


    def get_state_tensor(self):
        readingsX, readingsY = getPointCloud(lidar_sensors)
        state_tensor = torch.FloatTensor(getTensor(readingsX, readingsY, N_DIV))
        return state_tensor

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

#initialize robot sensors
gps = getGPS(robot, timestep)
dist_sensors = getDistSensors(robot, timestep)
lidar_sensors = getLidar(robot, timestep)
robot.step()

env = Environment(robot, TARGET, gps, dist_sensors, lidar_sensors)

#input for the network
n_actions = 3
state_tensor = env.get_state_tensor()
n_observations = len(state_tensor)

#initialize networks
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(100000)

steps_done = 0

def select_action(state):
    test = False
    random_action = True
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold or test == True:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            random_action = False
            policy_net_output = policy_net(state)
            print(f"policy net state: {policy_net_output}")
            print(f"max value index: {torch.argmax(policy_net_output)}")
            #return random_action, policy_net(state).max(1).indices.view(1, 1) #o que estava originalmente
            return random_action, torch.tensor([[torch.argmax(policy_net(state))]], device=device, dtype=torch.long)

    else:
        return random_action, torch.tensor([[random.choice(env.action_space)]], device=device, dtype=torch.long)

episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())



def optimize_model():
    print("Optimizing")

    if len(memory) < BATCH_SIZE:
        print("batch too small")
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))
    print(f"Batch:\n{batch}\n")

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)

    print(f"non final mask: {non_final_mask.shape}")


    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])

    #print(f"non final next states: {non_final_next_states.shape}")

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    # print(f"state shape: {state_batch.shape}, {state_batch.shape[1]}")
    # print(f"action shape: {action_batch.shape}")
    # print(f"reward shape: {reward_batch.shape}")
    #
    # print(f"state: {state_batch}")
    # print(f"action: {action_batch}")

    #action_batch = torch.repeat_interleave(action_batch, state_batch.shape[1], dim=0)
    #action_batch = action_batch[None, :]

    #print(f"changed action shape: {action_batch.shape}")

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = torch.max(target_net(non_final_next_states))
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    print("Optimization complete")

def train():

    if torch.cuda.is_available():
        print("using cuda")
        num_episodes = 500
    else:
        num_episodes = 50

    print(f"Running {num_episodes} episodes")

    save_rate = 10
    reward_hist = []

    for i_episode in range(num_episodes):

        #at the beggining of each episode reset env and get initial state
        env.reset()
        total_reward: float = 0
        state = env.get_state_tensor()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        print(f"Episode {i_episode} started")
        # Initialize the environment and get its state

        for t in count():

            #sample action and get resulting state
            print(f"Running episode {i_episode}")
            print(f"Timestep {t}")
            random_action, action = select_action(state)
            action_int = action.item()
            print(f"Action: {action_int}, random: {random_action}")
            print(f"Action: {env.action_dict[action_int]}")

            reward, truncated, terminated = env.step(action_int)

            next_state = env.get_state_tensor()
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
            print(f"Next State: {next_state}")

            if truncated:
                print("Truncated")
            if terminated:
                print("Terminated")
            print(f"Reward: {reward}")
            reward = torch.tensor([reward], device=device)
            total_reward+=float(reward)
            done = terminated or truncated

            if terminated:
                next_state = None

            # Store the transition state - action - next state - reward in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(t + 1)
                plot_durations()
                break

        if (i_episode + 1) % save_rate == 0 or i_episode == 0:
            torch.save({
                'episode': i_episode,
                'model_state_dict': policy_net.state_dict(),
                'optimizer_state_dict': policy_net.state_dict(),
            }, 'my_models/model_test_dqn_run' + str(i_episode + 1))

        reward_hist.append(total_reward)

    print('Complete')
    with open('reward_hist', 'wb') as f:
        pickle.dump(reward_hist, f)
    plot_durations(show_result=True)
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    train()