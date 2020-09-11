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

from joblib import Parallel, delayed


experiment_name = 'task_1'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)


# function that produces the offspring by combining the genomes
def crossover(remaining_solutions, population_size, num_to_kill_off, n_vars):
    children_to_spawn = num_to_kill_off

    # which parents to combine
    parent1 = np.random.randint(len(remaining_solutions) - num_to_kill_off, population_size, size=children_to_spawn)
    parent2 = np.random.randint(len(remaining_solutions) - num_to_kill_off, population_size, size=children_to_spawn)


    # create the genomes of the children by taking the first have of the weights from one partens
    # and the rest from another parent
    for i in range(children_to_spawn):

        remaining_solutions[i] = np.concatenate((remaining_solutions[parent1[i]][:int(n_vars/2)],
                                 remaining_solutions[parent2[i]][int(n_vars - n_vars/2):]))


    return remaining_solutions


# mutate one of the weights of a solution with a certain chance
def mutation(population, mutation_chance):
    for i in range(len(population)):
        if random.random() < mutation_chance:
            index_point_mutation = random.randint(0, len(population[i]))
            population[i][index_point_mutation] = -population[i][index_point_mutation]

    return population


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
    population_size = 50
    n_generations = 5
    mutation_chance = 0.2
    num_cores = mp.cpu_count()

    # create random population
    population_of_solutions = np.random.uniform(lower_bound, upper_bound, (population_size, n_vars))

    mean_fitness_per_generation = np.zeros(n_generations + 1)  # +1 for generation 0
    std_fitness_per_generation = np.zeros(n_generations + 1)  # +1 for generation 0

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

        # # the normal non parallelised loop
        # # play with the different solutions
        # for i in range(population_size):
        #     results = env.play(pcont=population_of_solutions[i])
        #     fitness, player_life, enemy_life, run_time = results
        #     fitnesses.append(fitness)

        mean_fitness = np.mean(fitnesses)
        mean_fitness_per_generation[gen_iter] = mean_fitness
        std_fitness_per_generation[gen_iter] = np.std(fitnesses)
        print('Generation: {}, with an average fitness: {}'.format(gen_iter, mean_fitness))

        # sort the population form worst to best
        sols_worst_to_best = np.array([x for _,x in sorted(list(zip(fitnesses, population_of_solutions)),
                                                           key=lambda x: x[0])])

        # remove the worst solutions
        num_to_kill_off = int(len(sols_worst_to_best)/2)
        sols_worst_to_best[0:num_to_kill_off] = 0

        # repopulate the solution pool by reproduction
        population_of_solutions = crossover(sols_worst_to_best, population_size, num_to_kill_off, n_vars)

        # randomly mutate some individuals
        population_of_solutions = mutation(population_of_solutions, mutation_chance)

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

    print('Final population solution has an average fitness of: {}'.format(np.mean(fitnesses)))
