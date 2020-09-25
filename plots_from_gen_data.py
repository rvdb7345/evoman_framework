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
from scipy.stats import ttest_ind, levene


pd.set_option('display.max_columns', 500)

### --------------------------------- results 1 --------------------------------------- ###

enemies = 8
model = 'roulette_weightedNpoint_adaptmutation'
name = 'robin'


# with open('gen_sum_' + model + '_' + name, 'rb') as config_dictionary_file:
#     config_df = pickle.load(config_dictionary_file)

with open('generational_summary_1', 'rb') as config_dictionary_file:
    config_df = pickle.load(config_dictionary_file)

print(config_df.head())

# enemy_specific_develop = config_df.loc[(config_df['enemies'] == enemies) & (config_df['model'] == model + '_' + name)]
enemy_specific_develop = config_df.loc[(config_df['enemies'] == enemies) & (config_df['model'] == model)]


mean_mean_per_generation_model1 = []
mean_std_per_generation_model1 = []

mean_max_per_generation_model1 = []
std_max_per_generation_model1 = []


for i, gen in enemy_specific_develop.groupby(by='gen'):
    mean_mean_per_generation_model1.append(gen['fit_mean'].mean())
    mean_std_per_generation_model1.append(gen['fit_mean'].std())

    mean_max_per_generation_model1.append(gen['fit_max'].mean())
    std_max_per_generation_model1.append(gen['fit_max'].std())

### --------------------------------- results 2 --------------------------------------- ###

model = 'distanceroulette_weightedNpoint_adaptmutation'
name = 'robin'


# with open('gen_sum_' + model + '_' + name, 'rb') as config_dictionary_file:
#     config_df = pickle.load(config_dictionary_file)

with open('generational_summary_1', 'rb') as config_dictionary_file:
    config_df = pickle.load(config_dictionary_file)

print(config_df.head())

# enemy_specific_develop = config_df.loc[(config_df['enemies'] == enemies) & (config_df['model'] == model + '_' + name)]
enemy_specific_develop = config_df.loc[(config_df['enemies'] == enemies) & (config_df['model'] == model)]


mean_mean_per_generation_model2 = []
mean_std_per_generation_model2 = []

mean_max_per_generation_model2 = []
std_max_per_generation_model2 = []


for i, gen in enemy_specific_develop.groupby(by='gen'):
    mean_mean_per_generation_model2.append(gen['fit_mean'].mean())
    mean_std_per_generation_model2.append(gen['fit_mean'].std())

    mean_max_per_generation_model2.append(gen['fit_max'].mean())
    std_max_per_generation_model2.append(gen['fit_max'].std())


num_gen = 120
f, a = plt.subplots()
plt.title('Fitness over generations, enemy: {}'.format(enemies))

# means
a_mean_ea1 = a.plot(mean_mean_per_generation_model1[:num_gen], label='Mean fitness - EA1', color='blue', linestyle='solid')
a_mean_ea2 = plt.plot(mean_mean_per_generation_model2[:num_gen], label='Mean fitness - EA2', color='orange', linestyle='dashed')

a_mean_fill_ea1 = a.fill_between(np.arange(0, len(mean_mean_per_generation_model1[:num_gen])),
                 np.array(mean_mean_per_generation_model1[:num_gen]) - np.array(mean_std_per_generation_model1[:num_gen]),
                 np.array(mean_mean_per_generation_model1[:num_gen])+np.array(mean_std_per_generation_model1[:num_gen]), alpha=0.3,
                 facecolor='blue', edgecolor="black", hatch="X")

a_mean_fill_ea2 = a.fill_between(np.arange(0, len(mean_mean_per_generation_model2[:num_gen])),
                 np.array(mean_mean_per_generation_model2[:num_gen]) - np.array(mean_std_per_generation_model2[:num_gen]),
                 np.array(mean_mean_per_generation_model2[:num_gen])+np.array(mean_std_per_generation_model2[:num_gen]), alpha=0.5,
                 facecolor='orange')

# maxes
a_max_ea1 = a.plot(mean_max_per_generation_model1[:num_gen], label='Max fitness - EA1', color='red', linestyle='solid')
a_max_ea2 = a.plot(mean_max_per_generation_model2[:num_gen], label='Max fitness - EA2', color='green', linestyle='dashed')


a_max_fill_ea1 = a.fill_between(np.arange(0, len(mean_max_per_generation_model1[:num_gen])),
                 np.array(mean_max_per_generation_model1[:num_gen]) - np.array(std_max_per_generation_model1[:num_gen]),
                 np.array(mean_max_per_generation_model1[:num_gen]) + np.array(std_max_per_generation_model1[:num_gen]), alpha=0.2,
                 facecolor='red', edgecolor="black", hatch="X")
a_max_fill_ea2 = a.fill_between(np.arange(0, len(mean_max_per_generation_model2[:num_gen])),
                 np.array(mean_max_per_generation_model2[:num_gen]) - np.array(std_max_per_generation_model2[:num_gen]),
                 np.array(mean_max_per_generation_model2[:num_gen]) + np.array(std_max_per_generation_model2[:num_gen]), alpha=0.6,
                 facecolor='green')

plt.legend([(a_mean_ea1[0], a_mean_fill_ea1), (a_mean_ea2[0], a_mean_fill_ea2),
            (a_max_ea1[0], a_max_fill_ea1), (a_max_ea2[0], a_max_fill_ea2)],
           ['Mean - EA1', 'Mean - EA2', 'Max - EA1', 'Max - EA2'], fontsize=12)
plt.xlabel('Generation (#)', fontsize=16)
plt.ylabel('Fitness', fontsize=16)
plt.xlim(0)

plt.savefig('generational_results_enemy{}.png'.format(enemies), dpi=300)
plt.show()

print("the average std dev of model1: {} and the average std dev of "
      "model2: {}".format(np.mean(mean_std_per_generation_model1), np.mean(mean_std_per_generation_model2)))
print('Levene test of the distributions: ', levene(mean_std_per_generation_model1, mean_std_per_generation_model2))
ttest_enemy_8 = ttest_ind(mean_std_per_generation_model1, mean_std_per_generation_model2)
print("a T-test of the standard deviation during the evolutionary run: ", ttest_enemy_8)

