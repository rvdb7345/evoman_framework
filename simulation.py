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
import pickle
import pandas as pd
from datetime import datetime
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
                multiplemode = "no",
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
        self.tau = 1 / self.pop_size ** 2

        if len(enemies) > 1 and multiplemode == "no" or len(enemies) == 1 and multiplemode == "yes":
            raise AssertionError("Enemies settings are not valid!")

        self.enemies = enemies
        self.multiplemode = "no"
        self.dir_path = os.path.abspath('')

        # initialize random AI players for given pop size
        self.pcontrols, self.controller_id = [], 0
        for i in range(pop_size):
            pcont = test_controller(
                self.controller_id, self.nn_topology, lb=self.lower_bound, ub=self.upper_bound
            )
            pcont.initialize_random_network()
            self.pcontrols.append(pcont)
            self.controller_id += 1

        # self.current_ids = list(range(pop_size))

        # initialize tracking variables for statistics of simulation
        self.mean_fitness_gens = np.zeros(nr_gens + 1)
        self.stds_fitness_gens = np.zeros(nr_gens + 1)
        self.mean_p_lifes_gens = np.zeros(nr_gens + 1)
        self.stds_p_lifes_gens = np.zeros(nr_gens + 1)

        self.mean_e_lifes_gens = np.zeros(nr_gens + 1)
        self.stds_e_lifes_gens = np.zeros(nr_gens + 1)


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

    def save_generations(self, generation_sum_df):
        if os.path.exists(os.path.join(self.dir_path, 'generational_summary')):
            with open(os.path.join(self.dir_path, 'generational_summary'), 'rb') as config_df_file:
                config_df = pickle.load(config_df_file)
                generation_sum_df = pd.concat([generation_sum_df, config_df])

        with open('generational_summary', 'wb') as config_dictionary_file:
            pickle.dump(generation_sum_df, config_dictionary_file)

    def save_best_solution(self, enemies, best_fit, sol):
        best_solution_df = pd.DataFrame({'enemies': enemies, 'fitness': best_fit, 'best_solution': sol}, index=[0])

        if os.path.exists(os.path.join(self.dir_path, 'best_results')):
            with open(os.path.join(self.dir_path, 'best_results'), 'rb') as config_df_file:
                config_df = pickle.load(config_df_file)
                best_solution_df = pd.concat([best_solution_df, config_df], ignore_index=True)

        with open('best_results', 'wb') as config_dictionary_file:
            pickle.dump(best_solution_df, config_dictionary_file)

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

    def mutation(self, child_params, str_layer):
        """
        Add standard normally distributed noise (mutate) to parameters child 
        """
        if np.random.uniform(0, 1) < self.mutation_chance:
            noise = np.random.normal(0, 1)
            child_params["W" + str_layer] += noise
            child_params["b" + str_layer] += noise

        return child_params

    def crossover(self, child_params, prob1, prob2, W1, W2, b1, b2, str_layer):
        """
        Linear weighted crossover of the weights in a certain layer
        """
        child_params["W" + str_layer] = prob1 * W1 + prob2 * W2
        child_params["b" + str_layer] = prob1 * b1 + prob2 * b2

        return child_params

    def crossover_division(self):
        """
        Determines the weight of the linear crossover based on a Standard
        Normal distributed probability 
        """
        prob1 = np.random.uniform(0, 1)
        prob2 = 1 - prob1

        return prob1, prob2

    def crossover_and_mutation(self, parent1, parent2):
        """
        Cross genes for each layer in hidden network by weighted linear combination.
        The probability is for now drawn from a uniform distribution (see make_new_generation)

        (according to a relative probability based on the fitnes of each parent??? --> 
        less diversity, cause a greedy approach --> so maybe not necessary?)
        """

        # setup variables for child controller and parametrs, nr of layer and
        # parameters of parents
        child_cont, child_params = test_controller(self.controller_id, self.nn_topology),  {}
        self.controller_id += 1
        network1, network2 = parent1.get_params(), parent2.get_params()

        # performs crossover per layer 
        for layer in range(self.tot_layers):
            str_layer = str(layer)

            # retrieve matrices parents and perform weighted linear combination
            W1, W2 = network1["W" + str_layer], network2["W" + str_layer]
            b1, b2 = network1["b" + str_layer], network2["b" + str_layer]
            activation_funcs = network1["activation" + str_layer], network2["activation" + str_layer]

            # performs crossover
            prob1, prob2 = self.crossover_division()
            child_params = self.crossover(child_params, prob1, prob2, W1, W2, b1, b2, str_layer)

            # determine activation function by same probabilities
            active_func = np.random.choice(activation_funcs, p=[prob1, prob2])
            child_params["activation" + str_layer] = active_func

            child_params = self.mutation(child_params, str_layer)

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

    def parent_selection(self, fit_norm, sorted_controls):
        """
        Perfrom one roulette rank selection based on the (linear) normalized 
        probabilites of the fitnesses (THIS ALSO COULD BE EXP)
        """
        pcont = np.random.choice(sorted_controls, p=fit_norm)
        idx = sorted_controls.index(pcont)
        return idx, pcont

    def determine_survival(self, fit_norm_sorted, sorted_controls, children):
        """
        """
        reversed_norm = [fit_norm_sorted[self.pop_size - i - 1] for i, _ in enumerate(fit_norm_sorted)]
        for _,  child in enumerate(children):
            id_parent, parent = self.parent_selection(reversed_norm, sorted_controls)
            sorted_controls[id_parent] = child

        return sorted_controls            

    def make_new_generation(self, fit_norm_sorted, sorted_controls):
        """
        Crossover gense for a given population
        """

        # start creating children based on pairs of parents
        children = []
        for i in range(0, self.pop_size, self.nr_skip_parents):

            # rank selection of two parents
            _, parent1 = self.parent_selection(fit_norm_sorted, sorted_controls)
            _, parent2 = self.parent_selection(fit_norm_sorted, sorted_controls)

            # create child and add to children list
            child = self.crossover_and_mutation(parent1, parent2)
            children.append(child)

        # select parents based on normilzed probabilites of the fitnesses who
        # do not survive
        sorted_controls = self.determine_survival(fit_norm_sorted, sorted_controls, children)

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
        player_lifes = [pool_list[i][1] for i in range(self.pop_size)]
        enemies_lifes = [pool_list[i][2] for i in range(self.pop_size)]

        best_fit_gen = max(fitnesses)
        if best_fit_gen > self.best_fit:
            self.best_fit = best_fit_gen
            self.best_sol = self.pcontrols[fitnesses.index(self.best_fit)]
            self.not_improved = 0
        else:
            self.not_improved += 1

        self.mean_fitness_gens[gen] = np.mean(fitnesses)
        self.stds_fitness_gens[gen] = np.std(fitnesses)

        self.mean_p_lifes_gens[gen] = np.mean(player_lifes)
        self.stds_p_lifes_gens[gen] = np.std(player_lifes)

        self.mean_e_lifes_gens[gen] = np.mean(enemies_lifes)
        self.stds_e_lifes_gens[gen] = np.std(enemies_lifes)

        return fitnesses, player_lifes

    def run_evolutionary_algo(self):
        """
        Run evolutionary algorithm in parallel
        """

        ## necessary probabilites for rank selection
        ranks = list(range(1, self.pop_size + 1, 1))
        sum_ranks = sum(ranks)
        fit_norm = [rank / sum_ranks for rank in ranks]

        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

        generation_sum_df = pd.DataFrame(columns=['datetime', 'gen', 'enemies', 'fit_max', 'fit_mean'])

        # start evolutionary algorithm
        for gen in tqdm(range(self.nr_gens)):

            fitnesses, player_lifes = self.run_parallel(gen)

            # create a (proportional) cdf for the fitnesses
            sorted_controls = [
                parent for _, parent in sorted(
                    list(zip(fitnesses, self.pcontrols)), key=lambda x: x[0]
                )
            ]
            fitnesses.sort()

            generation_sum_df = generation_sum_df.append(
                {'datetime': dt_string, 'gen': gen, 'enemies': self.enemies[0], 'fit_max': max(fitnesses),
                 'fit_mean': self.mean_fitness_gens[gen]}, ignore_index=True)

            # make new generation
            self.pcontrols = self.make_new_generation(fit_norm, sorted_controls)

        # run final solution in parallel
        fitnesses, player_lifes = self.run_parallel(self.nr_gens)

        # save the best solution of the entire run
        self.save_best_solution(self.enemies[0], self.best_fit, self.best_sol)

        # save the mean and the max fitness during each run
        self.save_generations(generation_sum_df)

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


class SimulationAdaptiveMutationNpointCrossover(SimulationRank):

    def mutation(self, child_params, str_layer, child_cont, parent1, parent2, W2, b2):
        # add noise (mutation)
        if np.random.random() < 0.5:
            mutation_step_size = parent1.mutation_step_size
        else:
            mutation_step_size = parent2.mutation_step_size

        mutation_step_size = mutation_step_size * np.exp(self.tau * np.random.normal(0, 1))

        child_cont.set_mutation_step_size(mutation_step_size)

        # add noise (mutation)
        for i in range(len(W2)):
            for j in range(len(W2[0])):
                if np.random.uniform(0, 1) < self.mutation_chance:
                    child_params["W" + str_layer][i][j] += mutation_step_size


        for i in range(np.shape(b2)[0]):
            if np.random.uniform(0, 1) < self.mutation_chance:
                child_params["b" + str_layer][i] += mutation_step_size

        return child_params

    def crossover(self, child_params, prob1, prob2, W1, W2, b1, b2, str_layer):
        child_params["W" + str_layer] = np.array([[W1[i][j] if np.random.random() < prob1 else W2[i][j]
                                                   for i in range(len(W2))] for j in range(len(W1[0]))]).T
        child_params["b" + str_layer] = np.array([b1[i] if np.random.random() < prob1 else b2[i]
                                                  for i in range(np.shape(b2)[0])])


        return child_params

    def crossover_division(self, fitnesses, id1, id2):
        prob1 = 0.0
        min_fitnesses = min(fitnesses)
        if min_fitnesses <= 0:
            make_positive = abs(min_fitnesses) + 1
            prob1 = (fitnesses[id1] + make_positive) / ((fitnesses[id2] + make_positive) +
                                                            (fitnesses[id1] + make_positive))
        else:
            prob1 = fitnesses[id1] / (fitnesses[id1] + fitnesses[id2])

        return prob1, 1 - prob1

    def crossover_and_mutation(self, parent1, parent2, id1, id2, fitnesses):
        """
        Cross genes for each layer in hidden network by weighted linear combination.
        The probability is for now drawn from a uniform distribution (see make_new_generation)

        (according to a relative probability based on the fitnes of each parent??? --> 
        less diversity, cause a greedy approach --> so maybe not necessary?)
        """

        # setup variables for child controller and parametrs, nr of layer and
        # parameters of parents
        child_cont, child_params = test_controller(self.controller_id, self.nn_topology),  {}
        self.controller_id += 1
        network1, network2 = parent1.get_params(), parent2.get_params()

        # performs crossover per layer 
        for layer in range(self.tot_layers):
            str_layer = str(layer)

            # retrieve matrices parents and perform weighted linear combination
            W1, W2 = network1["W" + str_layer], network2["W" + str_layer]
            b1, b2 = network1["b" + str_layer], network2["b" + str_layer]
            activation_funcs = network1["activation" + str_layer], network2["activation" + str_layer]

            # performs crossover
            prob1, prob2 = self.crossover_division(fitnesses, id1, id2)
            child_params = self.crossover(child_params, prob1, prob2, W1, W2, b1, b2, str_layer)

            # determine activation function by same probabilities
            active_func = np.random.choice(activation_funcs, p=[prob1, prob2])
            child_params["activation" + str_layer] = active_func

            child_params = self.mutation(child_params, str_layer, child_cont, parent1, parent2, W2, b2)

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

    def make_new_generation(self, fit_norm_sorted, sorted_controls, fitnesses):
        """
        Crossover gense for a given population
        """

        # start creating children based on pairs of parents
        children = []
        for i in range(0, self.pop_size, self.nr_skip_parents):

            # rank selection of two parents
            id1, parent1 = self.parent_selection(fit_norm_sorted, sorted_controls)
            id2, parent2 = self.parent_selection(fit_norm_sorted, sorted_controls)

            # create child and add to children list
            child = self.crossover_and_mutation(parent1, parent2, id1, id2, fitnesses)
            children.append(child)

        # select parents based on normilzed probabilites of the fitnesses who
        # do not survive
        sorted_controls = self.determine_survival(fit_norm_sorted, sorted_controls, children)

        return sorted_controls

    def run_evolutionary_algo(self):
        """
        Run evolutionary algorithm in parallel
        """

        ## necessary probabilites for rank selection
        ranks = list(range(1, self.pop_size + 1, 1))
        sum_ranks = sum(ranks)
        fit_norm = [rank / sum_ranks for rank in ranks]

        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

        generation_sum_df = pd.DataFrame(columns=['datetime', 'gen', 'enemies', 'fit_max', 'fit_mean'])

        # start evolutionary algorithm
        for gen in tqdm(range(self.nr_gens)):

            fitnesses, player_lifes = self.run_parallel(gen)

            # create a (proportional) cdf for the fitnesses
            sorted_controls = [
                parent for _, parent in sorted(
                    list(zip(fitnesses, self.pcontrols)), key=lambda x: x[0]
                )
            ]
            fitnesses.sort()

            generation_sum_df = generation_sum_df.append(
                {'datetime': dt_string, 'gen': gen, 'enemies': self.enemies[0], 'fit_max': max(fitnesses),
                 'fit_mean': self.mean_fitness_gens[gen]}, ignore_index=True)

            # make new generation
            self.pcontrols = self.make_new_generation(fit_norm, sorted_controls, fitnesses)

        # run final solution in parallel
        fitnesses, player_lifes = self.run_parallel(self.nr_gens)

        # save the best solution of the entire run
        self.save_best_solution(self.enemies[0], self.best_fit, self.best_sol)

        # save the mean and the max fitness during each run
        self.save_generations(generation_sum_df)

        # plot the results (mean and standard deviation) over the generations
        self.simple_errorbar()