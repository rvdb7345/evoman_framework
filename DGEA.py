################################
# EvoMan FrameWork - V1.0 2016 #
# Author: group 77             #
#                              #
################################

# imports framework
import sys, os
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller
import numpy as np
from tqdm import tqdm
import random
import multiprocessing as mp
from multiprocessing import Pool
import matplotlib.pyplot as plt
import math

from joblib import Parallel, delayed
os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = 'task_1'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)


def tournament_selection(sols, k, fitnesses):
    '''
        Selects the best individuals out of k individuals
    '''

    best = -1

    for i in range(k):
        idx = np.random.randint(0, len(sols))

        if best == -1 or fitnesses[idx] > fitnesses[best]:
            best = idx

    return sols[idx]

# function that produces the offspring by combining the genomes
def crossover(remaining_solutions, num_to_kill_off, fitnesses_remaining):
    '''
        produce offspring by doing a uniform-nonweighted crossover
    '''

    children_to_spawn = num_to_kill_off

    k = 20

    for i in range(children_to_spawn):
        parent1 = tournament_selection(remaining_solutions[num_to_kill_off:-1], k,
                                       fitnesses_remaining[num_to_kill_off:-1])
        parent2 = tournament_selection(remaining_solutions[num_to_kill_off:-1], k,
                                       fitnesses_remaining[num_to_kill_off:-1])

        child = [parent1[j] if np.random.random() < 0.5 else parent2[j] for j in range(len(parent1))]

        remaining_solutions[i] = child


    return remaining_solutions


# mutate one of the weights of a solution with a certain chance
def mutation(population, mutation_chance):

    # calculate the average individual
    average_joe = np.mean(population, axis=0)

    # mutate genes away from the average of the population
    for ind in population:
        for i in range(len(ind)):
            if np.random.random() < mutation_chance:
                ind[i] += np.random.normal(math.copysign(1, ind[i] - average_joe[i]) * 0.4, 1/np.random.random())

    return population


def calc_diversity(population):

    L = len(population[0]) * 2 ** 2 ** 0.5
    P = len(population)

    # calculate the average individual
    average_joe = np.mean(population, axis=0)


    diversity = 0
    for i in range(len(population)):

        # calculates the euclidian distance between two factors
        diversity += np.linalg.norm(population[i] - average_joe)

    diversity = 1 / (abs(L) * abs(P)) * diversity

    return diversity


# this function runs the simulation
def simulate(n_hidden_neurons, enemies, solution):
    # n_hidden_neurons = 10  # number of
    # enemies = [7, 8]  # enemies with which to test the solutions

    env = Environment(experiment_name=experiment_name,
                      enemies=enemies,
                      multiplemode="yes",
                      playermode="ai",
                      player_controller=player_controller(n_hidden_neurons),
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      logs="off")

    return env.play(pcont=solution)


if __name__ == '__main__':
    n_hidden_neurons = 10  # number of
    enemies = [7, 8]  # enemies with which to test the solutions

    # initialised once to get the n_vars
    env = Environment(experiment_name=experiment_name,
                      enemies=enemies,
                      multiplemode="yes",
                      playermode="ai",
                      player_controller=player_controller(n_hidden_neurons),
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      logs="off")

    env.state_to_log()  # checks environment state

    # the length of the input
    n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

    # set the parameters
    lower_bound = -1
    upper_bound = 1
    population_size = 25
    n_generations = 50
    mutation_chance = 0.75
    num_cores = mp.cpu_count()
    d_min = 5 * 10 ** (-6)
    d_high = 0.067
    fraction_to_replace = 0.2

    # create random population
    population_of_solutions = np.random.uniform(lower_bound, upper_bound, (population_size, n_vars))

    mean_fitness_per_generation = np.zeros(n_generations + 1)  # +1 for generation 0
    std_fitness_per_generation = np.zeros(n_generations + 1)  # +1 for generation 0
    diversity_per_generation = np.zeros(n_generations + 1)  # +1 for generation 0

    for gen_iter in tqdm(range(n_generations)):

        # create input including the number of neurons and the enemies so this isn't in the simulate function
        pool_input = [(n_hidden_neurons, enemies, sol) for sol in population_of_solutions]

        fitnesses = []

        # run the different solutions in parallel
        pool = Pool(mp.cpu_count())
        pool_list = pool.starmap(simulate, pool_input)
        pool.close()
        pool.join()

        # get the fitnesses from the total results formatted as [(f, p, e, t), (...), ...]
        fitnesses = [x[0] for x in pool_list]

        mean_fitness = np.mean(fitnesses)
        mean_fitness_per_generation[gen_iter] = mean_fitness
        std_fitness_per_generation[gen_iter] = np.std(fitnesses)

        print('Generation: {}, with an average fitness: {} + {}'.format(gen_iter, mean_fitness, np.std(fitnesses)))

        diversity_of_population = calc_diversity(population_of_solutions)

        diversity_per_generation[gen_iter] = diversity_of_population

        print("The current diversity of the population: ", diversity_of_population)

        # sort the population form worst to best
        sols_worst_to_best = np.array([x for _,x in sorted(list(zip(fitnesses, population_of_solutions)),
                                                           key=lambda x: x[0])])

        # remove the worst solutions
        num_to_kill_off = int(len(sols_worst_to_best)*fraction_to_replace)
        sols_worst_to_best[0:num_to_kill_off] = 0

        if diversity_of_population < d_high:
            print("mutating noww")
            population_of_solutions = mutation(population_of_solutions, mutation_chance)
        else:
            print("crossover now")
            population_of_solutions = crossover(sols_worst_to_best, num_to_kill_off, fitnesses)

        population_of_solutions = np.clip(population_of_solutions, lower_bound, upper_bound)


    # run final simulation with the end population
    pool_input = [(n_hidden_neurons, enemies, sol) for sol in population_of_solutions]

    fitnesses = []

    # run the different solutions in parallel
    pool = Pool(mp.cpu_count())
    pool_list = pool.starmap(simulate, pool_input)
    pool.close()
    pool.join()

    # get the fitnesses from the total results formatted as [(f, p, e, t), (...), ...]
    fitnesses = [x[0] for x in pool_list]

    # save final result
    mean_fitness = np.mean(fitnesses)
    mean_fitness_per_generation[n_generations] = mean_fitness
    std_fitness_per_generation[n_generations] = np.std(fitnesses)

    # plot the results over time
    plt.figure()
    plt.title('Fitness per generation')
    plt.errorbar(np.arange(0, n_generations + 1), mean_fitness_per_generation, yerr=std_fitness_per_generation)
    plt.grid()
    plt.xlabel('Generation (#)')
    plt.ylabel('Fitness')
    plt.show()

    plt.figure()
    plt.title('diversity')
    plt.plot(diversity_per_generation)
    plt.grid()
    plt.ylabel('diversity')
    plt.xlabel('generation')
    plt.show()

    print('Final population solution has an average fitness of: {}'.format(np.mean(fitnesses)))
