import numpy as np

reward_over_time = np.load('../reward_over_time_ppo_50_10.npy')
loss_over_time = np.load('../loss_over_time_ppo_50_10.npy')

print("REWARD")
print(reward_over_time)
print()
print("LOSS")
print(loss_over_time)