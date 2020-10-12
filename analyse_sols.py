#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A script to analyse the multiple "optimal" solutions found by the Monte Carlo
simulations in the EvoMan framework.

Created on Saturday Oct 10 2020
This code was implemented by
Julien Fer, Robin van den Berg, ...., ....
"""

# insert evoman framework to path
import sys
sys.path.insert(0, "evoman")

import os
import csv
import time
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind, levene, f_oneway

from environment import Environment
from demo_controller import player_controller

# set style sheet matplotlib
plt.style.use("seaborn-darkgrid")

def calc_diversity(population):
    """
    Determines the diversity by means of distance-to-average-point method
    """

    # determine the average vector, intialize diversity measure
    temp_pop = np.array(population)
    L = temp_pop[0].shape[0] * 2
    pop_size = temp_pop.shape[0]
    average_vec = np.mean(temp_pop, axis=0)
    diversity = 0

    for individual in temp_pop:
        diversity += np.linalg.norm(individual - average_vec)

    return (1 / (abs(L * pop_size))) * diversity

def find_maxdistance_ind(total_sols, individual, best_sols):
    """
    """

    # find solution with max distance and hopfully different behavior
    max_poss_distance = len(individual) * 2
    distances, max_distance, idx_max_distance = [], 0, 0
    for sol in range(total_sols):

        # determine distance and difference in fitness
        other_sol = best_sols[sol]
        distance = np.linalg.norm(individual - sol) / max_poss_distance

        # found "more different" solution
        if distance > max_distance:
            max_distance = distance
            idx_max_distance = sol

        distances.append(distance)
    
    # most "different" solution
    return best_sols[idx_max_distance]

def load_best_sols(experiment_name, filename_sols, filename_fits, N):
    """
    Loads best solutions with their corresponding best fitness of 
    Monte Carlo simulations into a ??? . Also determines the best solution and 
    the one that differs the most from the best solution.
    """
    relpath_sols = os.path.join("results", experiment_name, filename_sols)
    relpath_fits = os.path.join("results", experiment_name, filename_fits)
    
    # load solutions and their fitnesses
    best_sols = np.load(relpath_sols)
    total_sols = len(best_sols)
    best_fits = []
    with open(relpath_fits, 'r') as csv_file:
        reader = csv.reader(csv_file)
        new_best_sols = []
        for i, row in enumerate(reader):
            best_fits.append(float(row[0]))
            if i < total_sols:
                new_best_sols.append(best_sols["arr_{}".format(i)]) 
        best_fits = best_fits[1:]
        best_sols = new_best_sols

    # (bubble) sort solutions based on their fitness and select N best solutions
    i, swaps = 0, np.inf
    while (i < total_sols - 1 and swaps > 0):
        swaps = 0
        for j in range(i + 1, total_sols):
            if best_fits[i] > best_fits[j]:
                temp = best_fits[i]
                best_fits[i] = best_fits[j]
                best_fits[j] = temp

                temp = best_sols[i]
                best_sols[i] = best_sols[j]
                best_sols[j] = temp

                swaps += 1
        i += 1

    best_N_sols = best_sols[-N:]

    diversity = calc_diversity(best_N_sols)

    # find best solution and its most different one
    best_sol = best_N_sols[-1]
    sol_maxdistance = find_maxdistance_ind(total_sols, best_sol, best_sols)

    # most different solutions w.r.t. the best N solutions found
    most_different_sols = [find_maxdistance_ind(total_sols, ind, best_sols) for ind in best_N_sols]

    # returns best solution, most different solution and second best solution
    return best_sol, sol_maxdistance, best_N_sols, diversity, most_different_sols

def play_game(name, pcont, enemies, multiplemode, seed):
    """
    Helper function to simulate a game in the Evoman framework
    """
    np.random.seed(seed)
    env = Environment(
        experiment_name=name,
        enemies=enemies,
        multiplemode=multiplemode,
        playermode="ai",
        player_controller=player_controller(10),
        enemymode="static",
        level=2,
        speed="fastest",
        logs="off"
    )

    return env.play(pcont=pcont)

def run_solutions(name, N, solutions):
    """
    Run a certain solutions against all the enemies in the EvoMan framework
    """

    # run the given solutions against all enemies in parallel
    enemies = [1, 2, 3, 4, 5, 6, 7, 8]
    total_sols, total_enemies = len(solutions), len(enemies)
    seed, pool_input = 0, []
    for sol in solutions:
        for _ in range(N):
            for enemy in enemies:
                pool_input.append((name, sol, [enemy], "no", seed))
                seed += 1

    pool = Pool(cpu_count())
    pool_results = pool.starmap(play_game, pool_input)
    pool.close()
    pool.join()

    # get the statistics
    fitnesses, player_lifes, enemies_lifes = [], [], []
    for results in pool_results:
        fitness, player_life, enemy_life, _ = results
        fitnesses.append(fitness)
        player_lifes.append(player_life), enemies_lifes.append(enemy_life)
    
    # mean and max fitness per enemy
    column_per_enemy_fitness = np.reshape(fitnesses, (total_sols, N, total_enemies))
    mean_fitness_per_enemy = np.mean(column_per_enemy_fitness, axis=(0,1))
    max_fitness_per_enemy = np.amax(column_per_enemy_fitness, axis=(0,1))

    # determine individual gain for all runs
    player_lifes = np.reshape(player_lifes, (total_sols, N, total_enemies))
    enemies_lifes = np.reshape(enemies_lifes, (total_sols, N, total_enemies))
    column_per_enemy_indgain = player_lifes - enemies_lifes
    mean_indgain_per_sol_enemy = np.mean(column_per_enemy_indgain, axis=1)

    # mean total individual game per solution
    sum_indgain_per_sol = np.sum(column_per_enemy_indgain, axis=2)
    mean_indgain_per_sol = np.mean(sum_indgain_per_sol, axis=1)
    idx_best = np.argmax(mean_indgain_per_sol)
    best_solution = solutions[idx_best]
    best_indgain_per_enemy = mean_indgain_per_sol_enemy[idx_best, :]
    print("Best solution has index: ", np.argmax(mean_indgain_per_sol))

    print("This should be the average summed individual gain per solution: \n",
          mean_indgain_per_sol)

    # save mean indvidual gain and best gain per enemy of best solution 
    # and of course the best solution
    folder = os.path.join("results", "runs_best_solutions")
    os.makedirs(folder, exist_ok=True)
    replath = os.path.join(folder, name = "_mean_indgain_per_sol_per_enemy.npy")
    np.save(relpath, mean_indgain_per_sol_enemy)
    relpath = os.path.join(folder, name + "_mean_indgain_per_sol.npy")
    np.save(relpath, mean_indgain_per_sol)
    relpath = os.path.join(folder, name + "_bestgain.npy")
    np.save(relpath, best_solution)
    relpath = os.path.join(folder, name + "_bestgain_perenemy.npy")
    np.save(relpath, best_indgain_per_enemy)

    return column_per_enemy_fitness, column_per_enemy_indgain, mean_indgain_per_sol, best_solution, best_indgain_per_enemy

def statistical_tests(
        mean_indgain_dgea1, mean_indgain_dgea2, 
        mean_indgain_NB1, mean_indgain_NB2
    ):
    """
    Performs statistical test to determine if the two algorithms are the "same"
    aka equal distribution
    """
    levene1 = levene(mean_indgain_dgea1, mean_indgain_dgea2)
    ttest1 = ttest_ind(mean_indgain_dgea1, mean_indgain_dgea2)
    ftest1 = f_oneway(mean_indgain_dgea1, mean_indgain_dgea2)
    result1 = (levene1, ttest1, ftest1)

    levene2 = levene(mean_indgain_dgea1, mean_indgain_NB1)
    ttest2 = ttest_ind(mean_indgain_dgea1, mean_indgain_NB1)
    ftest2 = f_oneway(mean_indgain_dgea1, mean_indgain_NB1)
    result2 = (levene2, ttest2, ftest2)

    levene3 = levene(mean_indgain_NB1, mean_indgain_NB2)
    ttest3 = ttest_ind(mean_indgain_NB1, mean_indgain_NB2)
    ftest3 = f_oneway(mean_indgain_NB1, mean_indgain_NB2)
    result3 = (levene3, ttest3, ftest3)

    levene4 = levene(mean_indgain_dgea2, mean_indgain_NB2)
    ttest4 = ttest_ind(mean_indgain_dgea2, mean_indgain_NB2)
    ftest4 = f_oneway(mean_indgain_dgea2, mean_indgain_NB2)
    result4 = (levene4, ttest4, ftest4)

    # save statistics
    folder = os.path.join("results", "runs_best_solutions")
    os.makedirs(folder, exist_ok=True)
    relpath = os.path.join(folder, "statistics.txt")
    with open(relpath, 'w') as txt_file:
        results = [result1, result2, result3, result4]
        compares = ["DGEA 1 and DGEA 2", "DGEA 1 and NB 1", "NB 1 and NB 2", "DGEA 2 and NB 2"]
        for compare, result in zip(compares, results):
            levene_stat, ttest, ftest = result
            txt_file.write("Statistics between {}\n".format(compare))
            txt_file.write("Levene: {}\n".format(levene_stat))
            txt_file.write("T test: {}\n".format(ttest))
            txt_file.write("F test: {}\n".format(ftest))

    return result1, result2, result3, result4

def make_boxplot(
        mean_indgain_dgea1, mean_indgain_dgea2, 
        mean_indgain_NB1, mean_indgain_NB2
    ):
    """
    """
    folder = os.path.join("results", "runs_best_solutions")
    os.makedirs(folder, exist_ok=True)
    plt.figure(figsize=(10, 3))
    plt.boxplot([mean_indgain_dgea1, mean_indgain_dgea2, mean_indgain_NB1,
                 mean_indgain_NB2])
    plt.xticks([1, 2, 3, 4], ['DGEA - E78', 'DGEA - E26', 'NB - E78', 'NB - E26'], fontsize=14)
    plt.yticks(fontsize=14)
    plt.title("Individual gain per algorithm and training set")
    plt.ylabel('Individual Gain', fontsize=14)
    plt.savefig(os.path.join(folder, 'boxplot_best10solutions.pdf'), dpi=300)
    plt.show()

if __name__ == "__main__":
    reps, N_best = 5, 10
    filenames_dgea1 = ["dgea_dgea_first_e_set", "best_sols_e7e8.npz", "best_fits_best_solse7e8.csv"]
    filenames_dgea2 = ["dgea_dgea_second_e_set", "best_sols_e2e6.npz", "best_fits_best_solse2e6.csv"]
    filenames_NB1 = ["dgea_newblood_first_e_set", "best_sols_e7e8.npz", "best_fits_best_solse7e8.csv"]
    filenames_NB2 = ["dgea_newblood_second_e_set", "best_sols_e2e6.npz", "best_fits_best_solse2e6.csv"]

    print("Loading started")
    start_load = time.time()
    best_sol_dgea1, sol_maxdistance_dgea1, best_N_sols_dgea1, diversity_dgea1, most_different_sols_dgea1 = load_best_sols(*filenames_dgea1, N_best)
    best_sol_dgea2, sol_maxdistance_dgea2, best_N_sols_dgea2, diversity_dgea2, most_different_sols_dgea2 = load_best_sols(*filenames_dgea2, N_best)
    best_sol_NB1, sol_maxdistance_NB1, best_N_sols_NB1, diversity_NB1, most_different_sols_NB1 = load_best_sols(*filenames_NB1, N_best)
    best_sol_NB2, sol_maxdistance_NB2, best_N_sols_NB2, diversity_NB2, most_different_sols_NB2 = load_best_sols(*filenames_NB2, N_best)
    print("Loading finished in {} minutes".format(round((time.time() - start_load) / 60), 2))

    print("Run best solutions")
    start_run = time.time()
    results_dgea1 = run_solutions(filenames_dgea1[0], reps, best_N_sols_dgea1)
    _, _, mean_indgain_dgea1, best_solution_dgea1, best_indgain_dgea1 = results_dgea1

    results_dgea2 = run_solutions(filenames_dgea2[0], reps, best_N_sols_dgea2)
    _, _, mean_indgain_dgea2, best_solution_dgea2, best_indgain_dgea2 = results_dgea2

    results_NB1 = run_solutions(filenames_NB1[0], reps, best_N_sols_NB1)
    _, _, mean_indgain_NB1, best_solution_NB1, best_indgain_NB1 = results_NB1

    results_NB2 = run_solutions(filenames_NB2[0], reps, best_N_sols_NB2)
    _, _, mean_indgain_NB2, best_solution_NB2, best_indgain_NB2 = results_NB2
    print("Run best solutions finished in {} minutes".format(round((time.time() - start_run) / 60), 2))

    print("Determine statistics")
    start_stats = time.time()
    results_stats = statistical_tests(
        mean_indgain_dgea1, mean_indgain_dgea2, 
        mean_indgain_NB1, mean_indgain_NB2
    )
    print("Stats finished in {} minutes".format(round((time.time() - start_stats) / 60), 2))

    print("Make boxplot")
    start_box = time.time()
    make_boxplot(mean_indgain_dgea1, mean_indgain_dgea2, mean_indgain_NB1, mean_indgain_NB2)
    print("Boxplot finished in {} minutes".format(round((time.time() - start_box) / 60), 2))
    
    print("Total script ran in {} minutes".format(round((time.time() - start_load) / 60, 2)))

    # # run best solution ansd the most different one
    # results_best_diff_dgea1 = run_solutions("best_and_diff_dgea1", reps, [best_sol_dgea1, sol_maxdistance_dgea1])
    # _, _, mean_indgain_best_diff_dgea1, best_diff_solution_dgea1, best_diff_indgain_dgea1 = results_best_diff_dgea1

    # results_best_diff_dgea2 = run_solutions("best_and_diff_dgea2", reps, [best_sol_dgea2, sol_maxdistance_dgea2])
    # _, _, mean_indgain_best_diff_dgea2, best_diff_solution_dgea2, best_diff_indgain_dgea2 = results_best_diff_dgea2

    # results_best_diff_NB1 = run_solutions("best_and_diff_NB1", reps, [best_sol_NB1, sol_maxdistance_NB1])
    # _, _, mean_indgain_best_diff_NB1, best_diff_solution_NB1, best_diff_indgain_NB1 = results_best_diff_NB1

    # results_best_diff_NB2 = run_solutions("best_and_diff_NB2", reps, [best_sol_NB2, sol_maxdistance_NB2])
    # _, _, mean_indgain_best_diff_NB2, best_diff_solution_NB2, best_diff_indgain_NB2 = results_best_diff_NB2

