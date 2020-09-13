import pickle
import sys, os
sys.path.insert(0, "evoman")

from multiprocessing import Pool, cpu_count
from tqdm  import tqdm
import numpy as np
import matplotlib.pyplot as plt
from environment import Environment
from controller_julien import test_controller
from controller import Controller
import numpy as np


def run_one_parallel(
    pcontrols, enemies, pop_size, best_fit, gen,
    not_improved, mean_fitness_gens, stds_fitness_gens,
        mean_p_lifes_gens, stds_p_lifes_gens,
        mean_e_lifes_gens, stds_e_lifes_gens,
        best_sol
    ):
    """
    Runs one parralel simulation in the Evoman framework
    """

    # create input including the number of neurons and the enemies so this isn't in the simulate function
    pool_input = [(pcont, enemies, "no") for pcont in pcontrols]

    # run the simulations in parallel
    pool = Pool(cpu_count())
    pool_list = pool.starmap(play_game, pool_input)
    pool.close()
    pool.join()

    # get the fitnesses from the total results formatted as [(f, p, e, t), (...), ...]
    fitnesses = [pool_list[i][0] for i in range(pop_size)]
    player_lifes = [pool_list[i][1] for i in range(population_size)]
    enemies_lifes = [pool_list[i][2] for i in range(population_size)]

    best_fit_gen = max(fitnesses)
    if best_fit_gen > best_fit:
        best_fit = best_fit_gen
        best_sol = pcontrols[fitnesses.index(best_fit)]
        not_improved = 0
    else:
        not_improved += 1

    mean_fitness_gens[gen] = np.mean(fitnesses)
    stds_fitness_gens[gen] = np.std(fitnesses)

    mean_p_lifes_gens[gen] = np.mean(player_lifes)
    stds_p_lifes_gens[gen] = np.std(player_lifes)

    mean_e_lifes_gens[gen] = np.mean(enemies_lifes)
    stds_e_lifes_gens[gen] = np.std(enemies_lifes)

    return fitnesses, best_fit, player_lifes, enemies, best_sol


with open('best_results', 'rb') as config_dictionary_file:
    config_dictionary = pickle.load(config_dictionary_file)

    # After config_dictionary is read from file
    print(config_dictionary)