# import torch
# print(torch.cuda.is_available())

import numpy as np
mean, std, min, max = np.load('data/_energy_mean_std_min_max.npy')
min = (min - mean) / std
max = (max - mean) / std
np.save("./data/energy_mean_std_min_max.npy", np.array([mean, std, min, max]))
print([mean, std, min, max])
mean, std, min, max = np.load('data/_pitch_mean_std_min_max.npy')
min = (min - mean) / std
max = (max - mean) / std
np.save("./data/pitch_mean_std_min_max.npy", np.array([mean, std, min, max]))
print([mean, std, min, max])