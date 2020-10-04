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
from multiprocessing import Pool, cpu_count

# import third parties packages
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm as progressbar

# import custom module (others)
from environment import Environment
from demo_controller import player_controller
from sklearn.cluster import KMeans


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
        self.results_folder = os.path.join("results", name)
        self.n_hidden_neurons = parameters["neurons"]
        self.lower_bound, self.upper_bound = parameters["lb"], parameters["ub"]
        self.pop_size, self.total_generations = parameters["pop_size"], parameters["total_generations"]
        self.mutation_prob = parameters["mutation_prob"]
        self.mutation_frac = parameters["mutation_factor"] * abs(self.upper_bound - self.lower_bound)
        self.crossover_prob = parameters["crossover_prob"]
        self.dmin, self.dmax = parameters["dmin"], parameters["dmax"]
        self.fraction_replace = parameters["fraction_replace"]
        self.max_no_improvements = parameters["max_no_improvements"]
        self.enemies = enemies

        # initialize tracking variables for statistics of simulation
        self.fitnesses, self.best_fit_gens, self.diversity_gens = [], [], []
        self.best_fit, self.best_sol = None, None
        self.best_fits, self.best_sols = [], []
        self.not_improved = 0

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

        return (1 / (abs(self.L) * self.pop_size)) * diversity

    def tournament(self, population, fitnesses):
        """
        Performs Binary tournament.
        """
        idx1 = np.random.randint(0, population.shape[0])
        idx2 = np.random.randint(0, population.shape[0])

        if fitnesses[idx1] > fitnesses[idx2]:
            return population[idx1]

        return population[idx2]

    def crossover(self, population, fitnesses):
        """
        Performs hybrid crossover: uniform crossover and arithmetic crossover
        The uniform crossover randomly divides weights of 0 and 1 to the weights
        of the neurons. This is done for n-1 weights. The last one is drawn from
        a unifrom distribution. The weights are reversed for the second child.
        """
        nr_children = int(population.shape[0] * self.fraction_replace)
        all_offsprings = np.zeros((0, self.n_vars))
        all_offsprings = np.vstack((all_offsprings, self.best_sol))
        child = 1

        # start making offspring (two for each couple of parents)
        while child < nr_children:
            parent1 = self.tournament(population, fitnesses)
            parent2 = self.tournament(population, fitnesses)

            # determine weights
            weights = np.zeros(self.n_vars)
            weights[:self.n_vars - 1] = np.random.choice([0, 1], size=self.n_vars - 1)
            weights[self.n_vars - 1] = np.random.uniform()
            np.random.shuffle(weights)

            # perform crossover
            offspring1 = weights * parent1 + (1 - weights) * parent2
            offspring2 = (1 - weights) * parent1 + weights * parent2
            all_offsprings = np.vstack((all_offsprings, offspring1))
            all_offsprings = np.vstack((all_offsprings, offspring2))
            child += 2

        return all_offsprings

    def mutation(self, population, fitnesses=None):
        """
        Gaussian mutation operator with the mean directed away from the
        average point of the population
        """
        average_vec = np.mean(population, axis=0)
        scale_factors = np.ones(len(population[0])) * self.mutation_frac
        stds_gaussian = abs(self.upper_bound - self.lower_bound) / 2
        # stds_gaussian = abs(self.upper_bound - self.lower_bound) / 2 * 1/np.random.uniform()

        for individual in population:
            if np.random.uniform() < self.mutation_prob:
                distance_vec = individual - average_vec
                means_gaussian = np.copysign(scale_factors, distance_vec)
                individual += np.random.normal(means_gaussian, stds_gaussian)

        return population

    def update_statistics(self, curr_sim, gen, fitnesses, population, diversity):
        """
        Update the statistics for given generation
        """
        for fitness in fitnesses:
            stats = {
                "simulation": curr_sim,
                "generation": gen,
                "fitness": fitness
            }
            self.fitnesses.append(stats)

        best_fit_gen = max(fitnesses)
        self.best_fit_gens.append(
            {"simulation": curr_sim, "generation": gen, "best fit": best_fit_gen}
        )
        self.diversity_gens.append(
            {"simulation": curr_sim, "generation": gen, "diversity": diversity}
        )

        if self.best_fit is None or best_fit_gen > self.best_fit:
            self.best_fit = best_fit_gen
            self.best_sol = population[fitnesses.index(self.best_fit)]
            self.best_fits = [self.best_fit]
            self.best_sols = [self.best_sol]
            self.not_improved = 0

        # THIS FIRST NEED TO BE DISCUSSED BEFORE WE IMPLEMENT IT
        # we also should add a the diversity measure here cause otherwise 
        # we will save a lot of duplicate controllers with the same score instead
        # of different controllers with the (almost the?) same score
        elif best_fit_gen == self.best_fit:
            best_candidate = population[fitnesses.index(best_fit_gen)]
            distances = [np.linalg.norm(best_candidate - sol) for sol in self.best_sols]
            if min(distances) != 0:
                self.best_sols.append(best_candidate)
            
            if self.mode == "Explore":
                self.not_improved += 1

        # Only keep track of no improvements if we diversity is low (Explore fase)
        elif self.mode == "Explore":
            self.not_improved += 1

    def calc_clustering(self, population, curr_sim, gen):
        inertias = []
        clusters = np.arange(2, len(population))
        for i in clusters:
            kmeans = KMeans(n_clusters=i, random_state=0, n_jobs=-1).fit(population)
            inertias.append(kmeans.inertia_)

        enemies_str = ""
        for enemy in self.enemies:
            enemies_str += "e" + str(enemy)
        filename = "kmeans_clustering_sim" + str(curr_sim) + "_gen" + str(gen) + ".png"
        path = os.path.join(self.results_folder, filename)
        plt.figure()
        plt.plot(clusters, inertias)
        plt.xlabel('clusters (#)')
        plt.ylabel('Inertia')
        plt.title('Kmeans clustering of the population')
        plt.savefig(path, dpi=300)

    def run(self, curr_sim):
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

        # checks environment state
        env.state_to_log()

        # number of variables for neural network with one hidden layer
        self.n_vars = (env.get_num_sensors() + 1) * self.n_hidden_neurons + (self.n_hidden_neurons + 1) * 5

        # determine diagonal of search space (not sure if correct)
        self.L = math.sqrt(self.n_vars * (self.upper_bound - self.lower_bound) ** 2)

        # create initial random population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.n_vars))

        # initial mode dgea algorithm 
        gen, self.mode = 0, "Exploit"
        total_exploit, total_explore = 0, 0

        # initalize progressbar
        pbar = progressbar(total=self.total_generations, desc="evolutionary loop DGEA")

        # run initial population
        pool = Pool(cpu_count())
        pool_input = [sol for sol in population]
        pool_results = pool.map(self.play_game, pool_input)
        pool.close()
        pool.join()

        # save results inital solution
        fitnesses = [result[0] for result in pool_results]
        diversity = self.calc_diversity(population)
        self.update_statistics(curr_sim, gen, fitnesses, pool_input, diversity)

        pbar.update(1)

        # start evolutionary algorithm
        # for gen in progressbar(range(1, self.total_generations + 1)):
        while gen < self.total_generations + 1 and self.not_improved < self.max_no_improvements:
            gen += 1
            if diversity < self.dmin:
                if self.mode == "Exploit":
                    self.calc_clustering(population, curr_sim, gen)
                self.mode = "Explore"
            elif diversity > self.dmax:
                self.mode = "Exploit"

            if self.mode == "Explore":
                population = self.mutation(population, fitnesses)
                total_explore += 1
            elif self.mode == "Exploit":
                if np.random.uniform() < self.crossover_prob:
                    population = self.crossover(population, fitnesses)
                total_exploit += 1

            # make sure weights neural network are within borders
            population = np.clip(population, self.lower_bound, self.upper_bound)

            # run new population
            pool = Pool(cpu_count())
            pool_input = [sol for sol in population]
            pool_results = pool.map(self.play_game, pool_input)
            pool.close()
            pool.join()

            # save results inital solution
            fitnesses = [result[0] for result in pool_results]
            diversity = self.calc_diversity(population)
            self.update_statistics(curr_sim, gen, fitnesses, pool_input, diversity)

            pbar.update(1)

        self.calc_clustering(population, curr_sim, gen)
        pbar.close()

        values = [
            self.fitnesses, self.best_fit_gens, self.diversity_gens, self.best_fit,
            self.best_sol, self.best_sols, total_exploit, total_explore
        ]

        return values

    def reset_algorithm(self):
        """
        Resets algorithm so a new run can be performed
        """
        self.fitnesses, self.diversity_gens = [], []
        self.best_fit, self.best_sol = None, None
        self.best_sols, self.not_improved = [], 0


class NewBlood(DGEA):
    def mutation(self, population, fitnesses):
        '''
        This function replaces mutation_prob of the worst individuals of the population with random individuals. This
        was chosen to keep the number of individuals that are changed by a mutation step constant with the DGEA.
        '''

        # sort the population based on fitness and replace worst
        sorted_fit_pop = sorted(list(zip(fitnesses, population)), key=lambda x: x[0])
        sorted_pop = np.array([ind for _, ind in sorted_fit_pop])
        print(sorted_pop)
        ind_to_replace = int(self.mutation_prob * len(sorted_pop))
        sorted_pop[0:ind_to_replace] = 0

        # add new, random individuals to the population
        new_population = np.random.uniform(self.lower_bound, self.upper_bound, (ind_to_replace, self.n_vars))
        sorted_pop[0:ind_to_replace] = new_population

        return sorted_pop
