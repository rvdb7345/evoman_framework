################################
# EvoMan FrameWork - V1.0 2016 #
# Author: group 77             #
#                              #
################################

# imports framework
import sys, os
sys.path.insert(0, 'evoman')

# import built-in packages
import math
import random
from multiprocessing import Pool, cpu_count

# import third parties packages
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm as processbar

# import custom module (others)
from environment import Environment
from demo_controller import player_controller

class DGEA(object):
    """
    Standard form of Diversity Guides Evolutionary Algorithm (DGEA)
    """
    def __init__(
            self,
            name,
            parameters,
            enemies=[7, 8]
        ):
        self.name = name
        self.n_hidden_neurons = parameters["neurons"]
        self.lower_bound, self.upper_bound = parameters["lb"], parameters["ub"]
        self.pop_size, self.total_generations = parameters["pop_size"], parameters["total_generations"]
        self.mutation_prob = parameters["mutation_prob"]
        self.crossover_prob = parameters["crossover_prob"]
        self.dmin, self.dmax = parameters["dmin"], parameters["dmax"]
        self.fraction_replace = parameters["fraction_replace"]
        self.enemies = enemies

        # initialize tracking variables for statistics of simulation
        self.mean_fitness_gens = np.zeros(self.total_generations + 1)
        self.stds_fitness_gens = np.zeros(self.total_generations + 1)
        self.mean_p_lifes_gens = np.zeros(self.total_generations + 1)
        self.stds_p_lifes_gens = np.zeros(self.total_generations + 1)
        self.mean_e_lifes_gens = np.zeros(self.total_generations + 1)
        self.stds_e_lifes_gens = np.zeros(self.total_generations + 1)
        self.diversity_gens = np.zeros(self.total_generations + 1)
        self.best_fit, self.best_sol = None, None
        self.best_sols, self.not_improved = [], 0

    def play_game(self, sol):
        """
        Plays one game for a certain player controller
        """
        env = Environment(
            experiment_name=self.name,
            enemies=self.enemies,
            multiplemode="yes",
            playermode="ai",
            player_controller=player_controller(self.n_hidden_neurons),
            enemymode="static",
            level=2,
            speed="fastest",
            logs="off"
        )
        return env.play(pcont=sol)

    def calc_diversity(self, population):
        """
        Determines the diversity by means of distance-to-average-point method
        """

        # determine the average vector, intialize diversity measure
        average_vec = np.mean(population, axis=0)
        diversity = 0

        for individual in population:
            diversity += np.linalg.norm(individual - average_vec)

        return (1 / (abs(self.L) + self.pop_size)) * diversity

    def tournament(self, population, fitnesses):
        """
        Performs Binary tournament.
        """
        idx1 = np.random.randint(0, population.shape[0], 1)
        idx2 = np.random.randint(0, population.shape[0], 1)
        
        if fitnesses[idx1] > fitnesses[idx2]:
            return population[idx1]

        return fitnesses[idx2]

    def crossover(self, population, fitnesses):
        """
        Performs hybrid crossover: uniform crossover and arithmetic crossover
        The uniform crossover randomly divides weights of 0 and 1 to the weights
        of the neurons. This is done for n-1 weights. The last one is drawn from
        a unifrom distribution.
        """
        nr_children = int(population.shape[0] * self.fraction_replace)
        all_offsprings = np.zeros((0, self.n_vars))

        # start making offspring (two for each couple of parents)
        for child in range(0, nr_children, 2):
            parent1 = self.tournament(population, fitnesses)
            parent2 = self.tournament(population, fitnesses)

            # determine weights
            weights = np.zeros(self.n_vars)
            weights[:self.n_vars - 1] = np.random.choice([0, 1], size=self.n_vars-1)
            weights[self.n_vars - 1] = np.random.uniform()
            np.random.shuffle(weights)

            # perform crossover
            offspring1 = weights * parent1 + (1 - weights) * parent2
            offspring2 = (1 - weights) * parent1 + weights * parent2
            np.vstack((all_offsprings, offspring1)), np.vstack((all_offsprings, offspring2))

        return all_offsprings

    def update_statistics(self, gen, results, controls):
        """
        Update the statistics for given generation
        """
        fitnesses, player_lifes, enemies_lifes, time = [], [], [], []
        for result in results:
            fitnesses.append(result[0])
            player_lifes.append(result[1])
            enemies_lifes.append(result[2])
            time.append(result[3])

        self.mean_fitness_gens[gen] = np.mean(fitnesses)
        self.stds_fitness_gens[gen] = np.std(fitnesses)
        self.mean_p_lifes_gens[gen] = np.mean(player_lifes)
        self.stds_p_lifes_gens[gen] = np.std(player_lifes)
        self.mean_e_lifes_gens[gen] = np.mean(enemies_lifes)
        self.stds_e_lifes_gens[gen] = np.std(enemies_lifes)

        best_fit_gen = max(fitnesses)
        if self.best_fit is None or best_fit_gen > self.best_fit:
            self.best_fit = best_fit_gen
            self.best_sol = controls[fitnesses.index(self.best_fit)]
            self.best_fits = [self.best_fit]
            self.best_sols = [self.best_sol]
            self.not_improved = 0

        # we also should add a the diversity measure here cause otherwise 
        # we will save a lot of duplicate controllers with the same score instead
        # of different controllers with the same score
        elif best_fit_gen == self.best_fit:
            self.best_sols.append(controls[fitnesses.index(best_fit_gen)])
            self.not_improved += 1
        else:
            self.not_improved += 1
    
    def run(self):
        """
        Run evolutionary algorithm
        """

        # initialised once to get the n_vars
        env = Environment(
            experiment_name=self.name,
            enemies=self.enemies,
            multiplemode="yes",
            playermode="ai",
            player_controller=player_controller(self.n_hidden_neurons),
            enemymode="static",
            level=2,
            speed="fastest",
            logs="off"
        )

        env.state_to_log()  # checks environment state

        # number of variables for neural network with one hidden layer
        self.n_vars = (env.get_num_sensors() + 1) * self.n_hidden_neurons + (self.n_hidden_neurons + 1) * 5

        # determine diagonal of search space (NOT SURE IF CORRECT)
        self.L = self.n_vars * 2 ** 2 ** 0.5

        # create random population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, n_vars))

        # run initial population
        pool = Pool(cpu_count())
        pool_input = [sol for sol in population]
        pool_results = pool.map(self.play_game, pool_input)
        pool.close()
        pool.join()
        
        # save results inital solution
        self.update_statistics(0, pool_results, pool_input)

        # start evolutionary algorithm
        for gen in range(1, self.total_generations):
            pass

if __name__ == "__main__":
    print("Hello world")