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


experiment_name = 'task_1'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)


def crossover(remaining_solutions, population_size):
    children_to_spawn = population_size - len(remaining_solutions)

    parent1 = np.random.randint(0, len(remaining_solutions), size=children_to_spawn)
    parent2 = np.random.randint(0, len(remaining_solutions), size=children_to_spawn)

    for i in range(len(children_to_spawn)):
        remaining_solutions.append()

def mutation():
    # to be implement
    return



n_hidden_neurons = 10
enemies = [4, 5, 6, 7, 8]

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
                  logs="on")

# default environment fitness is assumed for experiment

env.state_to_log() # checks environment state

n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

lower_bound = -1
upper_bound = 1
population_size = 10
n_generations = 5

population_of_solutions = np.random.uniform(lower_bound, upper_bound, (population_size, n_vars))



for gen_iter in range(n_generations):
    fitnesses = []

    for i in tqdm(range(population_size)):
        results = env.play(pcont=population_of_solutions[i])
        fitness, player_life, enemy_life, run_time = results
        fitnesses.append(fitness)

    sols_worst_to_best = [x for _,x in sorted(zip(fitnesses,population_of_solutions))]

    # remove the worst solutions
    remaining_population = sols_worst_to_best[int(len(sols_worst_to_best)/2):-1]




print(fitnesses)
