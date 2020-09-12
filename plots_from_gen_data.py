import pickle
import sys, os
sys.path.insert(0, "evoman")

from multiprocessing import Pool, cpu_count
from tqdm  import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from environment import Environment
from controller_julien import test_controller
from controller import Controller
import numpy as np

# Step 2
with open('generational_summary', 'rb') as config_dictionary_file:
    # Step 3
    config_df = pickle.load(config_dictionary_file)

print(config_df)

enemies = 8

enemy_specific_develop = config_df.loc[config_df['enemies'] == enemies]

print(enemy_specific_develop)

mean_mean_per_generation = []
mean_std_per_generation = []

mean_max_per_generation = []
std_max_per_generation = []


for i, gen in enemy_specific_develop.groupby(by='gen'):
    mean_mean_per_generation.append(gen['fit_mean'].mean())
    mean_std_per_generation.append(gen['fit_mean'].std())

    mean_max_per_generation.append(gen['fit_max'].mean())
    std_max_per_generation.append(gen['fit_max'].std())

plt.figure()
plt.title('Fitness over generations, enemy: {}'.format(enemies))
plt.plot(mean_mean_per_generation[:25], label='Mean fitness', color='blue')
plt.fill_between(np.arange(0, len(mean_mean_per_generation[:25])),
                 np.array(mean_mean_per_generation[:25]) - np.array(mean_std_per_generation[:25]),
                 np.array(mean_mean_per_generation[:25])+np.array(mean_std_per_generation[:25]), alpha=0.5,
                 facecolor='blue')
plt.plot(mean_max_per_generation[:25], label='Max fitness', color='red')
plt.fill_between(np.arange(0, len(mean_max_per_generation[:25])),
                 np.array(mean_max_per_generation[:25]) - np.array(std_max_per_generation[:25]),
                 np.array(mean_max_per_generation[:25]) + np.array(std_max_per_generation[:25]), alpha=0.5,
                 facecolor='red')
plt.grid()
plt.legend(fontsize=18)
plt.xlabel('Generation (#)', fontsize=16)
plt.ylabel('Fitness', fontsize=16)
plt.show()

