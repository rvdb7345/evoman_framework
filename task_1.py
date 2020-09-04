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


n_hidden_neurons = 10  # number of
enemies = [7, 8]  # enemies with which to test the solutions

# initializes environment with ai player using random controller, playing against static enemy
# initializes simulation in multi evolution mode, for multiple static enemies.
env = Environment(experiment_name=experiment_name,
                  enemies=enemies,
                  multiplemode="yes",
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  logs="off")

# default environment fitness is assumed for experiment

env.state_to_log() # checks environment state

# the length of the input
n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

# set the parameters
lower_bound = -1
upper_bound = 1
population_size = 6
n_generations = 5
mutation_chance = 0.2

# create random population
population_of_solutions = np.random.uniform(lower_bound, upper_bound, (population_size, n_vars))


for gen_iter in tqdm(range(n_generations)):
    fitnesses = []

    # play with the different solutions
    for i in range(population_size):
        results = env.play(pcont=population_of_solutions[i])
        fitness, player_life, enemy_life, run_time = results
        fitnesses.append(fitness)

    # sort the population form worst to best
    sols_worst_to_best = np.array([x for _,x in sorted(list(zip(fitnesses, population_of_solutions)), key=lambda x: x[0])])

    # remove the worst solutions
    num_to_kill_off = int(len(sols_worst_to_best)/2)
    sols_worst_to_best[0:num_to_kill_off] = 0

    # repopulate the solution pool by reproduction
    population_of_solutions = crossover(sols_worst_to_best, population_size, num_to_kill_off, n_vars)

    # randomly mutate some individuals
    population_of_solutions = mutation(population_of_solutions, mutation_chance)

    print('Generation: {}, with an average fitness: {}'.format(gen_iter, np.mean(fitnesses)))

# play with the final solutions
for i in range(population_size):
    results = env.play(pcont=population_of_solutions[i])
    fitness, player_life, enemy_life, run_time = results
    fitnesses.append(fitness)

print('Final population solution has an average fitness of: {}'.format(np.mean(fitnesses)))
