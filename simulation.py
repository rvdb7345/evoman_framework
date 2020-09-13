################################################################################
# Representation of a parallel simulation with as task to optimize an AI       #
# player, with help of evolutionary algorithm for the Evoman framerwork. There #
# are multiple version with variations in their crossover, mutation and        #
# selection methods.                                                           #
# 
# Names:
# University:
#                                                                              
################################################################################

# imports framework
import sys, os
sys.path.insert(0, "evoman")

from multiprocessing import Pool, cpu_count
from tqdm  import tqdm
import numpy as np
import matplotlib.pyplot as plt
from environment import Environment
from controller_julien import test_controller

class SimulationRank(object):
    """
    Basic form of siulation with linear rank selection for parents, uniformly 
    weighted crossover, standard normally distributed bias, truncated selection
    for durvival.
    """
    def __init__(self, 
                experiment_name="basic_simulation", 
                nr_inputs=20, 
                nr_layers=1, 
                nr_neurons=10, 
                nr_outputs=5,
                activation_func=["sigmoid"], 
                activation_distr=[1],
                lower_bound=-1, 
                upper_bound=1, 
                pop_size=10, 
                nr_gens=5, 
                mutation_chance=0.2, 
                nr_skip_parents=4, 
                enemies=[8], 
                multiplemode = "no"
        ):
        """
        Initialize simulation object
        """

        # set experiment name and make neural network topology
        self.name = experiment_name
        if not os.path.exists(experiment_name):
            os.makedirs(experiment_name)
        self.tot_layers = nr_layers + 1
        self.nn_topology = []
        self.make_NN_topology(
            nr_inputs, nr_layers, nr_neurons, 
            nr_outputs, activation_func, activation_distr
        )

        # set remaining attributes and make sure they are valid
        self.pop_size = pop_size
        self.nr_gens = nr_gens

        assert lower_bound <= upper_bound, "lower bound is greater than upper bound"
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.mutation_chance = mutation_chance
        self.nr_skip_parents = nr_skip_parents

        if len(enemies) > 1 and multiplemode == "no" or len(enemies) == 1 and multiplemode == "yes":
            raise AssertionError("Enemies settings are not valid!")

        self.enemies = enemies
        self.multiplemode = "no"

        # initialize random AI players for given pop size
        self.pcontrols = []
        for _ in range(pop_size):
            pcont = test_controller(
                self.nn_topology, lb=self.lower_bound, ub=self.upper_bound
            )
            pcont.initialize_random_network()
            self.pcontrols.append(pcont)

        # initialize tracking variables for statistics of simulation
        self.mean_fitness_gens = np.zeros(nr_gens + 1)
        self.stds_fitness_gens = np.zeros(nr_gens + 1)
        self.best_fit, self.best_sol, self.not_improved = 0, None, 0
        
    def make_NN_topology(
                self, nr_inputs, nr_layers, nr_neurons, 
                nr_outputs, activation_func, activation_distr
        ):
        """
        Determines the topology of the neural network. If multiple (hidden) 
        layers then it assums that all have the same amount of neurons.
        """
        for layer in range(self.tot_layers):
            layer_topology = {}

            #  inputs layer
            if layer == 0:
                layer_topology["input_dim"] = nr_inputs
                layer_topology["output_dim"] = nr_neurons

            # outputs layer
            elif layer == nr_layers:
                layer_topology["input_dim"] = nr_neurons
                layer_topology["output_dim"] = nr_outputs

            # hidden layers
            else:
                layer_topology["input_dim"] = nr_neurons
                layer_topology["output_dim"] = nr_neurons

            # activation function current layer
            layer_topology["activation"] = np.random.choice(
                activation_func, p=activation_distr
            )

            self.nn_topology.append(layer_topology)

    def play_game(self, pcont):
        """
        Helper function to simulate a game in the Evoman framework
        """
        env = Environment(
            experiment_name=self.name,
            enemies=self.enemies,
            multiplemode=self.multiplemode,
            playermode="ai",
            player_controller=pcont,
            enemymode="static",
            level=2,
            speed="fastest",
            logs="off"
        )

        return env.play(pcont=pcont)

    def crossover_and_mutation(self, parent1, parent2):
        """
        Cross genes for each layer in hidden network by weighted linear combination.
        The probability is for now drawn from a uniform distribution (see make_new_generation)

        (according to a relative probability based on the fitnes of each parent??? --> 
        less diversity, cause a greedy approach --> so maybe not necessary?)
        """

        # setup variables for child controller and parametrs, nr of layer and
        # parameters of parents
        child_cont, child_params = test_controller(self.nn_topology),  {}
        network1, network2 = parent1.get_params(), parent2.get_params()

        # performs crossover per layer 
        for layer in range(self.tot_layers):
            str_layer = str(layer)

            # retrieve matrices parents and perform weighted linear combination
            W1, W2 = network1["W" + str_layer], network2["W" + str_layer]
            b1, b2 = network1["b" + str_layer], network2["b" + str_layer]
            activation_funcs = network1["activation" + str_layer], network2["activation" + str_layer]

            # determine (uniform) probability of genes inherited by parents
            prob1 = np.random.uniform(0, 1)
            prob2 = 1 - prob1
            child_params["W" + str_layer] = prob1 * W1 + prob2 * W2
            child_params["b" + str_layer] = prob1 * b1 + prob2 * b2

            # determine activation function by same probabilities
            active_func = np.random.choice(activation_funcs, p=[prob1, prob2])
            child_params["activation" + str_layer] = active_func

            # add noise (mutation)
            if np.random.uniform(0, 1) < self.mutation_chance:
                noise = np.random.normal(0, 1)
                child_params["W" + str_layer] += noise
                child_params["b" + str_layer] += noise

            # adjust for limits weights
            weights_child = child_params["W" + str_layer]
            bias_child = child_params["b" + str_layer]
            weights_child[weights_child > self.upper_bound] = self.upper_bound
            weights_child[weights_child < self.lower_bound] = self.lower_bound
            bias_child[bias_child > self.upper_bound] = self.upper_bound
            bias_child[bias_child < self.lower_bound] = self.lower_bound

        # create network and return child
        child_cont.create_network(child_params)
        return child_cont

    def parent_selection(self, fit_norm, sorted_controls, id_prev=-1):
        """
        Perfrom one roulette rank selection based on the (linear) normalized 
        probabilites of the fitnesses (THIS ALSO COULD BE EXP)
        """
        pcont = np.random.choice(sorted_controls, p=fit_norm)
        idx = sorted_controls.index(pcont)
        if id_prev == idx and idx + 1 < self.pop_size:
            return idx, sorted_controls[idx + 1]
        
        return idx, pcont

    def make_new_generation(self, fit_norm_sorted, sorted_controls):
        """
        Crossover gense for a given population
        """

        # start creating childrens by pairs (for only a quarter of the population)
        # truncated "killing" selection
        children = []
        for i in range(0, self.pop_size, self.nr_skip_parents):

            # rank selection of two parents
            id1, parent1 = self.parent_selection(fit_norm_sorted, sorted_controls, id_prev=-1)
            id2, parent2 = self.parent_selection(fit_norm_sorted, sorted_controls, id_prev=-1)

            # create child and add to children list
            child = self.crossover_and_mutation(parent1, parent2)
            children.append(child)

        # replace the parents with the lowest score with the newly made children
        # and update population (truncation selection)
        sorted_controls[0:len(children)] = children

        return sorted_controls
        
    def run_parallel(self, gen):
        """
        Runs one parralel simulation in the Evoman framework
        """

        # run the simulations in parallel
        pool = Pool(cpu_count())
        pool_list = pool.map(self.play_game, self.pcontrols)
        pool.close()
        pool.join()

        # get the fitnesses from the total results formatted as [(f, p, e, t), (...), ...]
        fitnesses = [pool_list[i][0] for i in range(self.pop_size)]

        best_fit_gen = max(fitnesses)
        if best_fit_gen > self.best_fit:
            self.best_fit = best_fit_gen
            self.best_sol = self.pcontrols[fitnesses.index(self.best_fit)]
            self.not_improved = 0
        else:
            self.not_improved += 1

        self.mean_fitness_gens[gen] = np.mean(fitnesses)
        self.stds_fitness_gens[gen] = np.std(fitnesses)

        return fitnesses

    def run_evolutionary_algo(self):
        """
        Run evolutionary algorithm in parallel
        """

        ## necessary probabilites for rank selection
        ranks = list(range(1, self.pop_size + 1, 1))
        sum_ranks = sum(ranks)
        fit_norm = [rank / sum_ranks for rank in ranks]

        # start evolutionary algorithm
        for gen in tqdm(range(self.nr_gens)):

            fitnesses = self.run_parallel(gen)

            # create a (proportional) cdf for the fitnesses
            sorted_controls = [
                parent for _, parent in sorted(
                    list(zip(fitnesses, self.pcontrols)), key=lambda x: x[0]
                )
            ]
            fitnesses.sort()

            # make new generation
            self.pcontrols = self.make_new_generation(fit_norm, sorted_controls)

        # run final solution in parallel
        fitnesses = self.run_parallel(self.nr_gens)

        # plot the results (mean and standard deviation) over the generations
        self.simple_errorbar()
    
    def simple_errorbar(self):
        """
        Default matplotlib errorbar for mean fitnees
        """
        plt.figure()
        plt.title("Fitness per generation")
        plt.errorbar(
            np.arange(0, self.nr_gens + 1), self.mean_fitness_gens, yerr=self.stds_fitness_gens
        )
        plt.grid()
        plt.xlabel("Generation (#)")
        plt.ylabel("Fitness")
        plt.show()
    
class SimulationRoulette(SimulationRank):
    def __init__(
            self, experiment_name, nr_inputs, nr_layers, 
            nr_neurons, nr_outputs, activation_func, activation_distr,
            lower_bound, upper_bound, pop_size, nr_gens, 
            mutation_chance, nr_skip_parents, enemies,  multiplemode
        ):
        super().__init__(
            experiment_name, nr_inputs, nr_layers, nr_neurons, 
            nr_outputs, activation_func, activation_distr,
            lower_bound, upper_bound, pop_size, nr_gens, 
            mutation_chance, nr_skip_parents, enemies,  multiplemode
        )

    def run_evolutionary_algo(self):
        """
        Run evolutionary algorithm in parallel
        """

        # start evolutionary algorithm
        for gen in tqdm(range(self.nr_gens)):

            fitnesses = self.run_parallel(gen)

            # create a (proportional) cdf for the fitnesses
            sorted_controls = [
                parent for _, parent in sorted(
                    list(zip(fitnesses, self.pcontrols)), key=lambda x: x[0]
                )
            ]
            fitnesses.sort()

            # normalize fitness values to represent probabilites for 
            # roulette wheel selection
            best_fit_gen, worst_fit_gen = max(fitnesses), min(fitnesses)
            fit_norm = []
            if worst_fit_gen <= 0:
                adjusted_fits = np.array(fitnesses) + abs(worst_fit_gen) + 1
                sum_fits = adjusted_fits.sum()
                fit_norm = [fit / sum_fits for fit in adjusted_fits]
            else:
                sum_fits = sum(fitnesses)
                fit_norm = [fit / sum_fits for fit in fitnesses]

            # make new generation
            self.pcontrols = self.make_new_generation(fit_norm, sorted_controls)

        # run final solution in parallel
        fitnesses = self.run_parallel(self.nr_gens)

        # plot the results (mean and standard deviation) over the generations
        self.simple_errorbar()