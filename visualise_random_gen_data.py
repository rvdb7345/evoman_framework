import pickle
import sys, os
sys.path.insert(0, "evoman")

from multiprocessing import Pool, cpu_count
import matplotlib.patches as mpatches
from tqdm  import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from environment import Environment
from controller_julien import test_controller
from controller import Controller
import numpy as np

pd.set_option('display.max_columns', 500)

### --------------------------------- results 1 --------------------------------------- ###

enemies = 8
model = 'random_weightedNpoint_adapt'
name = 'robin'


# Step 2
with open('gen_sum_' + model + '_' + name, 'rb') as config_dictionary_file:
    # Step 3
    config_df = pickle.load(config_dictionary_file)

# with open('generational_summary_1', 'rb') as config_dictionary_file:
#     # Step 3
#     config_df = pickle.load(config_dictionary_file)

print(config_df.head())

enemy_specific_develop = config_df.loc[(config_df['enemies'] == enemies) & (config_df['model'] == model + '_' + name)]
# enemy_specific_develop = config_df.loc[(config_df['enemies'] == enemies) & (config_df['model'] == model)]


mean_mean_per_generation_model1 = []
mean_std_per_generation_model1 = []

mean_max_per_generation_model1 = []
std_max_per_generation_model1 = []


for i, gen in enemy_specific_develop.groupby(by='gen'):
    mean_mean_per_generation_model1.append(gen['fit_mean'].mean())
    mean_std_per_generation_model1.append(gen['fit_mean'].std())

    mean_max_per_generation_model1.append(gen['fit_max'].mean())
    std_max_per_generation_model1.append(gen['fit_max'].std())

num_gen = 120
f, a = plt.subplots()
plt.title('Fitness over generations, enemy: {}'.format(enemies))

# means
a_mean_ea1 = a.plot(mean_mean_per_generation_model1[:num_gen], label='Mean fitness - EA1', color='blue', linestyle='solid')

a_mean_fill_ea1 = a.fill_between(np.arange(0, len(mean_mean_per_generation_model1[:num_gen])),
                 np.array(mean_mean_per_generation_model1[:num_gen]) - np.array(mean_std_per_generation_model1[:num_gen]),
                 np.array(mean_mean_per_generation_model1[:num_gen])+np.array(mean_std_per_generation_model1[:num_gen]), alpha=0.3,
                 facecolor='blue', edgecolor="black", hatch="X")

# maxes
a_max_ea1 = a.plot(mean_max_per_generation_model1[:num_gen], label='Max fitness - EA1', color='red', linestyle='solid')


a_max_fill_ea1 = a.fill_between(np.arange(0, len(mean_max_per_generation_model1[:num_gen])),
                 np.array(mean_max_per_generation_model1[:num_gen]) - np.array(std_max_per_generation_model1[:num_gen]),
                 np.array(mean_max_per_generation_model1[:num_gen]) + np.array(std_max_per_generation_model1[:num_gen]), alpha=0.2,
                 facecolor='red', edgecolor="black", hatch="X")

plt.legend([(a_mean_ea1[0], a_mean_fill_ea1),
            (a_max_ea1[0], a_max_fill_ea1)],
           ['Mean - EA1', 'Max - EA1'], fontsize=12)
plt.xlabel('Generation (#)', fontsize=16)
plt.ylabel('Fitness', fontsize=16)
plt.xlim(0)

plt.savefig('generational_results_enemy{}.png'.format(enemies), dpi=300)
plt.show()

