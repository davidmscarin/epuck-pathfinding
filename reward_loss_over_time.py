import numpy as np

REWARD_PATH = '../reward_over_time_ppo_50_10.npy'
LOSS_PATH = '../loss_over_time_ppo_50_10.npy'

reward_over_time = np.load(REWARD_PATH)
loss_over_time = np.load(LOSS_PATH)

print("REWARD")
print(reward_over_time)
print()
print("LOSS")
print(loss_over_time)