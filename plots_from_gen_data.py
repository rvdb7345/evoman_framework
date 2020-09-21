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

pd.set_option('display.max_columns', 500)

enemies = 8
model = 'roulette_weightedNpoint_adapt'
name = 'robin'


# Step 2
with open('gen_sum_' + model + '_' + name, 'rb') as config_dictionary_file:
    # Step 3
    config_df = pickle.load(config_dictionary_file)

print(config_df.head())

enemy_specific_develop = config_df.loc[(config_df['enemies'] == enemies) & (config_df['model'] == model + '_' + name)]

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

num_gen = 80
plt.figure()
plt.title('Fitness over generations, enemy: {}, \n model: {}'.format(enemies, model))
plt.plot(mean_mean_per_generation[:num_gen], label='Mean fitness', color='blue')
plt.fill_between(np.arange(0, len(mean_mean_per_generation[:num_gen])),
                 np.array(mean_mean_per_generation[:num_gen]) - np.array(mean_std_per_generation[:num_gen]),
                 np.array(mean_mean_per_generation[:num_gen])+np.array(mean_std_per_generation[:num_gen]), alpha=0.5,
                 facecolor='blue')
plt.plot(mean_max_per_generation[:num_gen], label='Max fitness', color='red')
plt.fill_between(np.arange(0, len(mean_max_per_generation[:num_gen])),
                 np.array(mean_max_per_generation[:num_gen]) - np.array(std_max_per_generation[:num_gen]),
                 np.array(mean_max_per_generation[:num_gen]) + np.array(std_max_per_generation[:num_gen]), alpha=0.5,
                 facecolor='red')
plt.grid()
plt.legend(fontsize=18)
plt.xlabel('Generation (#)', fontsize=16)
plt.ylabel('Fitness', fontsize=16)
plt.show()

