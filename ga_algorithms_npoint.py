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
import sys
sys.path.insert(0, "evoman")

# import built-in packages
# import pickle
from multiprocessing import Pool, cpu_count
from datetime import datetime

# import third-party packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# import own modules
from environment import Environment
from controller_julien import test_controller

class GA_random_Npoint(object):
    """
    Basic (random) form of Genetic Algorithm. Note every selection, crossover, 
    mutation and survival method is based on random chance. The only method that
    is fixed is the crossover (random N-point crossover)
    """
    def __init__(self,
                name = "random_randomNpoint_normal",
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
                replacement = False
        ):
        # make neural network topology
        self.name = name
        # if not os.path.exists(experiment_name):
        #     os.makedirs(experiment_name)
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
        self.multiplemode = multiplemode
        self.replacement = replacement
        # self.show_plot = show_plot
        # self.save_output = save_output
        # self.dir_path = os.path.abspath('')

        # print("Do we try to save the output? : ", self.save_output)

        # initialize random AI players for given pop size
        self.pcontrols, self.controller_id = [], 0
        self.initialize_controllers()

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

    def initialize_controllers(self):
        """
        Initialize random AI players for given pop size
        """
        self.pcontrols, self.controller_id = [], 0
        for i in range(self.pop_size):
            pcont = test_controller(
                self.controller_id, self.nn_topology, lb=self.lower_bound, ub=self.upper_bound
            )
            pcont.initialize_random_network()
            self.pcontrols.append(pcont)
            self.controller_id += 1

    # def save_generations(self, generation_sum_df):
    #     if os.path.exists(os.path.join(self.dir_path, 'generational_summary')):
    #         with open(os.path.join(self.dir_path, 'generational_summary'), 'rb') as config_df_file:
    #             config_df = pickle.load(config_df_file)
    #             generation_sum_df = pd.concat([generation_sum_df, config_df])

    #     with open('generational_summary', 'wb') as config_dictionary_file:
    #         pickle.dump(generation_sum_df, config_dictionary_file)

    # def save_best_solution(self, enemies, best_fit, sol):
    #     best_solution_df = pd.DataFrame({'model': self.name, 'enemies': enemies,
    #                                      'fitness': best_fit, 'best_solution': sol}, index=[0])

    #     if os.path.exists(os.path.join(self.dir_path, 'best_results')):
    #         with open(os.path.join(self.dir_path, 'best_results'), 'rb') as config_df_file:
    #             config_df = pickle.load(config_df_file)
    #             best_solution_df = pd.concat([best_solution_df, config_df], ignore_index=True)

    #     with open('best_results', 'wb') as config_dictionary_file:
    #         pickle.dump(best_solution_df, config_dictionary_file)

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

    def crossover_division(self):
        """
        Determines the probality for the N-point crossover (two parents)
        """
        prob1 = np.random.uniform(0, 1)
        prob2 = 1 - prob1

        return prob1, prob2

    def crossover(self, child_params, prob1, prob2, W1, W2, b1, b2, str_layer):
        """
        Performs an N-point crossover by random probability determined by uniform
        distribution, so that on average prob1% is inherited from parent1 and
        (1 - prob1)% from parent2
        """
        child_params["W" + str_layer] = np.array([[W1[i][j] if np.random.random() < prob1 else W2[i][j]
                                                   for i in range(len(W2))] for j in range(len(W1[0]))]).T
        child_params["b" + str_layer] = np.array([b1[i] if np.random.random() < prob1 else b2[i]
                                                  for i in range(np.shape(b2)[0])])

        return child_params

    def mutation(self, child_params, str_layer):
        """
        Add standard normally distributed noise (mutate) to parameters child 
        """
        if np.random.uniform(0, 1) < self.mutation_chance:
            noise = np.random.normal(0, 1)
            child_params["W" + str_layer] += noise
            child_params["b" + str_layer] += noise

        return child_params

    def crossover_and_mutation(self, parent1, parent2):
        """
        Cross genes for each layer in hidden network by random N-point crossover combination.
        The probability is drawn from a uniform distribution 
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

    def parent_selection(self):
        """
        Randomly selects a parent
        """
        pcont = np.random.choice(self.pcontrols)
        idx = self.pcontrols.index(pcont)
        return idx, pcont

    def determine_survival(self, children):
        """
        Randomly replace parents by children. Note that two different methods
        are possible: with replacement of child or without replacement
        """
        if not self.replacement:

            # first select parents
            id_parents = []
            for _, child in enumerate(children):
                id_parent, _ = self.parent_selection()
                id_parents.append(id_parent)

            # replace parents by children
            for id_parent, child in zip(id_parents, children):
                self.pcontrols[id_parent] = child
            
            return self.pcontrols

        # with replacement of children in the pool of survival selection
        for _,  child in enumerate(children):
            id_parent, parent = self.parent_selection(sorted_controls)
            self.pcontrols[id_parent] = child

        return self.pcontrols

    def make_new_generation(self):
        """
        Crossover gense for a given population
        """

        # start creating children based on pairs of parents
        children = []
        for i in range(0, self.pop_size, self.nr_skip_parents):

            # random selection of two parents
            _, parent1 = self.parent_selection()
            _, parent2 = self.parent_selection()

            # create child and add to children list
            child = self.crossover_and_mutation(parent1, parent2)
            children.append(child)

        # randomly determines survival and replacement
        sorted_controls = self.determine_survival(children)

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
        fitnesses, player_lifes, enemies_lifes = [], [], []
        for score in pool_list:
            fitnesses.append(score[0])
            player_lifes.append(score[1])
            enemies_lifes.append(score[2])

        # fitnesses = [pool_list[i][0] for i in range(self.pop_size)]
        # player_lifes = [pool_list[i][1] for i in range(self.pop_size)]
        # enemies_lifes = [pool_list[i][2] for i in range(self.pop_size)]

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

        return fitnesses, player_lifes, enemies_lifes

    def run_evolutionary_algo(self):
        """
        Run evolutionary algorithm in parallel
        """

        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

        generation_sum_df = pd.DataFrame(columns=['datetime', 'gen', 'enemies', 'fit_max', 'fit_mean'])

        # start evolutionary algorithm
        for gen in tqdm(range(self.nr_gens)):

            fitnesses, player_lifes, enemies_lifes = self.run_parallel(gen)

            generation_sum_df = generation_sum_df.append(
                {'model': self.name, 'datetime': dt_string, 'gen': gen, 'enemies': self.enemies[0], 'fit_max': max(fitnesses),
                 'fit_mean': self.mean_fitness_gens[gen]}, ignore_index=True)

            # make new generation
            self.pcontrols = self.make_new_generation()

        # run final solution in parallel
        fitnesses, player_lifes, enemies_lifes = self.run_parallel(self.nr_gens)

        generation_sum_df = generation_sum_df.append(
            {'model': self.name, 'datetime': dt_string, 'gen': self.nr_gens, 'enemies': self.enemies[0],
             'fit_max': max(fitnesses),
             'fit_mean': self.mean_fitness_gens[gen]}, ignore_index=True)

        # returns certain statistics  (watch out self.enemies is hardcoded)
        return self.enemies[0], self.best_fit, self.best_sol, generation_sum_df

        # # save the best solution of the entire run save the mean and
        # # the max fitness during each run
        # if self.save_output:
        #     self.save_best_solution(self.enemies[0], self.best_fit, self.best_sol)
        #     self.save_generations(generation_sum_df)

        # # plot the results (mean and standard deviation) over the generations
        # if self.show_plot:
        #     self.simple_errorbar()
    
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

    def reset_algorithm(self):
        """
        Resets tracking variables so that a new simulation can be started
        """
        self.initialize_controllers()
        self.mean_fitness_gens = np.zeros(self.nr_gens + 1)
        self.stds_fitness_gens = np.zeros(self.nr_gens + 1)
        self.mean_p_lifes_gens = np.zeros(self.nr_gens + 1)
        self.stds_p_lifes_gens = np.zeros(self.nr_gens + 1)
        self.mean_e_lifes_gens = np.zeros(self.nr_gens + 1)
        self.stds_e_lifes_gens = np.zeros(self.nr_gens + 1)
        self.best_fit, self.best_sol, self.not_improved = 0, None, 0

class GA_roulette_randomNpoint(GA_random_Npoint):
    """
    Roulette wheel selection method for reproduction and survival, N-point crossover 
    and mutation are still random
    """
    def parent_selection(self, fit_norm, sorted_controls):
        """
        Perfrom one roulette fitness selection based on the (linear) normalized 
        probabilites of the fitnesses (THIS ALSO COULD BE EXP)
        """
        pcont = np.random.choice(sorted_controls, p=fit_norm)
        idx = sorted_controls.index(pcont)
        return idx, pcont

    def determine_survival(self, fit_norm_sorted, sorted_controls, children):
        """
        Replace parents based on their normilized fitness by the newly made children. 
        Note that two different methods are possible: with replacement of child 
        or without replacement
        """
        reversed_norm = [fit_norm_sorted[self.pop_size - i - 1] for i, _ in enumerate(fit_norm_sorted)]

        # without replacement of children in pool of survival
        if not self.replacement:
            id_parents = []
            for _, child in enumerate(children):
                id_parent, _ = self.parent_selection(reversed_norm, sorted_controls)
                id_parents.append(id_parent)

            for id_parent, child in zip(id_parents, children):
                sorted_controls[id_parent] = child

            return sorted_controls

        # with replacement of children in pool of survival
        for _, child in enumerate(children):
            id_parent, _ in self.parent_selection(reversed_norm, sorted_controls)
            sorted_controls[id_parent] = child

        return sorted_controls            

    def make_new_generation(self, fit_norm_sorted, sorted_controls):
        """
        Crossover gense for a given population
        """

        # start creating children based on pairs of parents
        children = []
        for i in range(0, self.pop_size, self.nr_skip_parents):

            # roulette wheel selection of two parents
            _, parent1 = self.parent_selection(fit_norm_sorted, sorted_controls)
            _, parent2 = self.parent_selection(fit_norm_sorted, sorted_controls)

            # create child and add to children list
            child = self.crossover_and_mutation(parent1, parent2)
            children.append(child)

        # select parents based on normilzed probabilites of the fitnesses who
        # do not survive
        sorted_controls = self.determine_survival(fit_norm_sorted, sorted_controls, children)

        return sorted_controls

    def run_evolutionary_algo(self):
        """
        Run evolutionary algorithm in parallel
        """

        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

        generation_sum_df = pd.DataFrame(columns=['datetime', 'gen', 'enemies', 'fit_max', 'fit_mean'])

        # start evolutionary algorithm
        for gen in tqdm(range(self.nr_gens)):

            fitnesses, player_lifes, enemies_lifes = self.run_parallel(gen)

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

            generation_sum_df = generation_sum_df.append(
                {'model': self.name, 'datetime': dt_string, 'gen': gen, 'enemies': self.enemies[0], 'fit_max': max(fitnesses),
                 'fit_mean': self.mean_fitness_gens[gen]}, ignore_index=True)

            # make new generation
            self.pcontrols = self.make_new_generation(fit_norm, sorted_controls)

        # run final solution in parallel
        fitnesses, player_lifes, enemies_lifes = self.run_parallel(self.nr_gens)

        generation_sum_df = generation_sum_df.append(
            {'model': self.name, 'datetime': dt_string, 'gen': self.nr_gens, 'enemies': self.enemies[0],
             'fit_max': max(fitnesses),
             'fit_mean': self.mean_fitness_gens[gen]}, ignore_index=True)

        # returns certain statistics  (watch out self.enemies is hardcoded)
        return self.enemies[0], self.best_fit, self.best_sol, generation_sum_df

        # # save the best solution of the entire run save the mean and
        # # the max fitness during each run
        # if self.save_output:
        #     self.save_best_solution(self.enemies[0], self.best_fit, self.best_sol)
        #     self.save_generations(generation_sum_df)

        # plot the results (mean and standard deviation) over the generations
        # if self.show_plot:
        #     self.simple_errorbar()

class GA_roulette_randomNpoint_scramblemutation(GA_roulette_randomNpoint):
    """
    Roulette wheel selection method for reproduction and survival, random N-point
    crossover and scramble mutation
    """
    def mutation(self, child_params, str_layer):
        """
        Performs scramble mutation by random shuffling a set of weights of the
        neural network for the child
        """

        # scramble the child's parameters for mutation
        start_point = random.randint(1, 3)
        end_point = random.randint(start_point, len(child_params["W" + str_layer]))
        
        for i in range(start_point, end_point):
            np.random.shuffle(child_params["W" + str_layer][i])                             
                
        return child_params

class GA_roulette_randomNpoint_adaptmutation(GA_roulette_randomNpoint):
    """
    Roulette wheel selection method for reproduction and survival, random N-point
    crossover and self-adaptive mutation
    """

    def mutation(self, child_params, str_layer, child_cont, parent1, parent2, W2, b2):
        """
        Performs self-adaptive mutation
        """
        
        # determine mutation step size that will be used from one of the parents
        if np.random.random() < 0.5:
            mutation_step_size = parent1.mutation_step_size
        else:
            mutation_step_size = parent2.mutation_step_size

        # get new mutation step size and set child's mutation step size equal to it
        mutation_step_size = mutation_step_size * np.exp(self.tau * np.random.normal(0, 1))
        child_cont.set_mutation_step_size(mutation_step_size)

        # add noise (mutation) for the weights
        for i in range(len(W2)):
            for j in range(len(W2[0])):
                if np.random.uniform(0, 1) < self.mutation_chance:
                    child_params["W" + str_layer][i][j] += mutation_step_size

        # add noise (mutation) for the biases
        for i in range(np.shape(b2)[0]):
            if np.random.uniform(0, 1) < self.mutation_chance:
                child_params["b" + str_layer][i] += mutation_step_size

        return child_params

    def crossover_and_mutation(self, parent1, parent2):
        """
        Cross genes for each layer in hidden network by random N-point crossover combination.
        The probability is drawn from a uniform distribution 
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

class GA_roulette_weightedNpoint(GA_roulette_randomNpoint):
    """
    Roulette wheel selection method for reproduction and survival, weighted 
    N-point crossover, mutation is random
    """

    def crossover_division(self, fitnesses, id1, id2):
        """
        Determines weighted probability for N-point crossover
        """
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
        Cross genes for each layer in hidden network by weighted N-point crossover
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

    def make_new_generation(self, fit_norm_sorted, sorted_controls, fitnesses):
        """
        Crossover gense for a given population
        """

        # start creating children based on pairs of parents
        children = []
        for i in range(0, self.pop_size, self.nr_skip_parents):

            # roulette wheel selection of two parents
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

        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

        generation_sum_df = pd.DataFrame(columns=['datetime', 'gen', 'enemies', 'fit_max', 'fit_mean'])


        # start evolutionary algorithm
        for gen in tqdm(range(self.nr_gens)):

            fitnesses, player_lifes, enemies_lifes = self.run_parallel(gen)

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

            generation_sum_df = generation_sum_df.append(
                {'model': self.name, 'datetime': dt_string, 'gen': gen, 'enemies': self.enemies[0],
                 'fit_max': max(fitnesses),
                 'fit_mean': self.mean_fitness_gens[gen]}, ignore_index=True)

            # make new generation
            self.pcontrols = self.make_new_generation(fit_norm, sorted_controls, fitnesses)

        # run final solution in parallel
        fitnesses, player_lifes, enemies_lifes = self.run_parallel(self.nr_gens)

        generation_sum_df = generation_sum_df.append(
            {'model': self.name, 'datetime': dt_string, 'gen': self.nr_gens, 'enemies': self.enemies[0],
             'fit_max': max(fitnesses),
             'fit_mean': self.mean_fitness_gens[gen]}, ignore_index=True)

        # returns certain statistics  (watch out self.enemies is hardcoded)
        return self.enemies[0], self.best_fit, self.best_sol, generation_sum_df

        # # save the best solution of the entire run save the mean and
        # # the max fitness during each run
        # if self.save_output:
        #     self.save_best_solution(self.enemies[0], self.best_fit, self.best_sol)
        #     self.save_generations(generation_sum_df)


        # plot the results (mean and standard deviation) over the generations
        # if self.show_plot:
        #     self.simple_errorbar()

class GA_roulette_weightedNpoint_scramblemutation(GA_roulette_weightedNpoint):
    """
    Roulette wheel selection method for reproduction and survival, weighted N-point
    crossover and scramble mutation
    """
    def mutation(self, child_params, str_layer):
        """
        Performs scramble mutation by random shuffling a set of weights of the
        neural network for the child
        """

        # scramble the child's parameters for mutation
        start_point = random.randint(1, 3)
        end_point = random.randint(start_point, len(child_params["W" + str_layer]))
        
        for i in range(start_point, end_point):
            np.random.shuffle(child_params["W" + str_layer][i])                             
                
        return child_params

class GA_roulette_weightedNpoint_adaptmutation(GA_roulette_weightedNpoint):
    """
    Roulette wheel selection method for reproduction and survival, weighted 
    N-point crossover, self-adaptive  mutation
    """
    
    def mutation(self, child_params, str_layer, child_cont, parent1, parent2, W2, b2):
        """
        Performs self-adaptive mutation
        """
        
        # determine mutation step size that will be used from one of the parents
        if np.random.random() < 0.5:
            mutation_step_size = parent1.mutation_step_size
        else:
            mutation_step_size = parent2.mutation_step_size

        # get new mutation step size and set child's mutation step size equal to it
        mutation_step_size = mutation_step_size * np.exp(self.tau * np.random.normal(0, 1))
        child_cont.set_mutation_step_size(mutation_step_size)

        # add noise (mutation) for the weights
        for i in range(len(W2)):
            for j in range(len(W2[0])):
                if np.random.uniform(0, 1) < self.mutation_chance:
                    child_params["W" + str_layer][i][j] += mutation_step_size

        # add noise (mutation) for the biases
        for i in range(np.shape(b2)[0]):
            if np.random.uniform(0, 1) < self.mutation_chance:
                child_params["b" + str_layer][i] += mutation_step_size

        return child_params

    def crossover_and_mutation(self, parent1, parent2, id1, id2, fitnesses):
        """
        Cross genes for each layer in hidden network by weighted N-point crossover
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

class GA_roulette_weightedNpoint_adaptscramblemutation(GA_roulette_weightedNpoint_adaptmutation):
    """
    Roulette wheel selection method for reproduction and survival, weighted 
    N-point crossover, self-adaptive mutation combined with a scramble mutation
    """
    def mutation(self, child_params, str_layer, child_cont, parent1, parent2, W2, b2):
        """
        Performs self-adaptive mutation
        """
        
        # determine mutation step size that will be used from one of the parents
        if np.random.random() < 0.5:
            mutation_step_size = parent1.mutation_step_size
        else:
            mutation_step_size = parent2.mutation_step_size

        # get new mutation step size and set child's mutation step size equal to it
        mutation_step_size = mutation_step_size * np.exp(self.tau * np.random.normal(0, 1))
        child_cont.set_mutation_step_size(mutation_step_size)

        # add noise (mutation) for the weights
        for i in range(len(W2)):
            for j in range(len(W2[0])):
                if np.random.uniform(0, 1) < self.mutation_chance:
                    child_params["W" + str_layer][i][j] += mutation_step_size

        # add noise (mutation) for the biases
        for i in range(np.shape(b2)[0]):
            if np.random.uniform(0, 1) < self.mutation_chance:
                child_params["b" + str_layer][i] += mutation_step_size

        # scramble the child's parameters for mutation
        start_point = random.randint(1, 3)
        end_point = random.randint(start_point, len(child_params["W" + str_layer]))   
        for i in range(start_point, end_point):
            np.random.shuffle(child_params["W" + str_layer][i])                             
                
        return child_params

class GA_distanceroulette_randomNpoint(GA_roulette_randomNpoint):
    """
    Roulette wheel selection with a heuristc based on the relative distance to 
    all parents, random N-point crossover and mutiation
    """

    def get_distance_probs(self, parent1, sorted_controls):
        """
        Determines the  probability based on the distance of the first
        parent selected relatively to that of all the others.
        """
        distances, params1 = [], parent1.get_params()
        for other_parent in sorted_controls:
            params2 = other_parent.get_params()
            distance = 0
            
            for layer in range(self.tot_layers):
                weight_str = "W" + str(layer)
                W1, W2 = params1[weight_str], params2[weight_str]
                diff = (W1 - W2).flatten()
                distance += np.sqrt(np.dot(diff, diff))

            distances.append(distance)

        # normalize, so we get probabilities
        sum_distances = sum(distances)
        distances_norm = [distance / sum_distances for distance in distances]

        return distances_norm

    def make_new_generation(self, fit_norm_sorted, sorted_controls):
        """
        Crossover gense for a given population
        """

        # start creating children based on pairs of parents
        children = []
        for i in range(0, self.pop_size, self.nr_skip_parents):

            # roulette wheel selection of two parents with the second
            # parent selected with help of distance heuristic
            _, parent1 = self.parent_selection(fit_norm_sorted, sorted_controls)
            distances_norm = self.get_distance_probs(parent1, sorted_controls)
            _, parent2 = self.parent_selection(distances_norm, sorted_controls)

            # create child and add to children list
            child = self.crossover_and_mutation(parent1, parent2)
            children.append(child)

        # select parents based on normilzed probabilites of the fitnesses who
        # do not survive
        sorted_controls = self.determine_survival(fit_norm_sorted, sorted_controls, children)

        return sorted_controls

class GA_distanceroulette_randomNpoint_scramblemutation(GA_roulette_randomNpoint_scramblemutation):
    """
    Roulette wheel selection with a heuristc based on the relative distance to 
    all parents, weighted N-point crossover and random mutiation
    """

    def get_distance_probs(self, parent1, sorted_controls):
        """
        Determines the  probability based on the distance of the first
        parent selected relatively to that of all the others.
        """
        distances, params1 = [], parent1.get_params()
        for other_parent in sorted_controls:
            params2 = other_parent.get_params()
            distance = 0
            
            for layer in range(self.tot_layers):
                weight_str = "W" + str(layer)
                W1, W2 = params1[weight_str], params2[weight_str]
                diff = (W1 - W2).flatten()
                distance += np.sqrt(np.dot(diff, diff))

            distances.append(distance)

        # normalize, so we get probabilities
        sum_distances = sum(distances)
        distances_norm = [distance / sum_distances for distance in distances]

        return distances_norm

    def make_new_generation(self, fit_norm_sorted, sorted_controls):
        """
        Crossover gense for a given population
        """

        # start creating children based on pairs of parents
        children = []
        for i in range(0, self.pop_size, self.nr_skip_parents):

            # roulette wheel selection of two parents with the second
            # parent selected with help of distance heuristic
            _, parent1 = self.parent_selection(fit_norm_sorted, sorted_controls)
            distances_norm = self.get_distance_probs(parent1, sorted_controls)
            _, parent2 = self.parent_selection(distances_norm, sorted_controls)

            # create child and add to children list
            child = self.crossover_and_mutation(parent1, parent2)
            children.append(child)

        # select parents based on normilzed probabilites of the fitnesses who
        # do not survive
        sorted_controls = self.determine_survival(fit_norm_sorted, sorted_controls, children)

        return sorted_controls

class GA_distanceroulette_randomNpoint_adaptmutation(GA_roulette_randomNpoint_adaptmutation):
    """
    Roulette wheel selection with a heuristc based on the relative distance to 
    all parents, random N-point crossover and self-adaptive mutiation
    """

    def get_distance_probs(self, parent1, sorted_controls):
        """
        Determines the  probability based on the distance of the first
        parent selected relatively to that of all the others.
        """
        distances, params1 = [], parent1.get_params()
        for other_parent in sorted_controls:
            params2 = other_parent.get_params()
            distance = 0
            
            for layer in range(self.tot_layers):
                weight_str = "W" + str(layer)
                W1, W2 = params1[weight_str], params2[weight_str]
                diff = (W1 - W2).flatten()
                distance += np.sqrt(np.dot(diff, diff))

            distances.append(distance)

        # normalize, so we get probabilities
        sum_distances = sum(distances)
        distances_norm = [distance / sum_distances for distance in distances]

        return distances_norm

    def make_new_generation(self, fit_norm_sorted, sorted_controls):
        """
        Crossover gense for a given population
        """

        # start creating children based on pairs of parents
        children = []
        for i in range(0, self.pop_size, self.nr_skip_parents):

            # roulette wheel selection of two parents with the second
            # parent selected with help of distance heuristic
            _, parent1 = self.parent_selection(fit_norm_sorted, sorted_controls)
            distances_norm = self.get_distance_probs(parent1, sorted_controls)
            _, parent2 = self.parent_selection(distances_norm, sorted_controls)

            # create child and add to children list
            child = self.crossover_and_mutation(parent1, parent2)
            children.append(child)

        # select parents based on normilzed probabilites of the fitnesses who
        # do not survive
        sorted_controls = self.determine_survival(fit_norm_sorted, sorted_controls, children)

        return sorted_controls

class GA_distanceroulette_weightedNpoint(GA_roulette_weightedNpoint):
    """
    Roulette wheel selection with a heuristc based on the relative distance to 
    all parents, weighted N-point crossover and random mutiation
    """

    def get_distance_probs(self, parent1, sorted_controls):
        """
        Determines the  probability based on the distance of the first
        parent selected relatively to that of all the others.
        """
        distances, params1 = [], parent1.get_params()
        for other_parent in sorted_controls:
            params2 = other_parent.get_params()
            distance = 0
            
            for layer in range(self.tot_layers):
                weight_str = "W" + str(layer)
                W1, W2 = params1[weight_str], params2[weight_str]
                diff = (W1 - W2).flatten()
                distance += np.sqrt(np.dot(diff, diff))

            distances.append(distance)

        # normalize, so we get probabilities
        sum_distances = sum(distances)
        distances_norm = [distance / sum_distances for distance in distances]

        return distances_norm

    def make_new_generation(self, fit_norm_sorted, sorted_controls, fitnesses):
        """
        Crossover gense for a given population
        """

        # start creating children based on pairs of parents
        children = []
        for i in range(0, self.pop_size, self.nr_skip_parents):

            # roulette wheel selection of two parents and second parent selected
            # with help of the distance heuristic
            id1, parent1 = self.parent_selection(fit_norm_sorted, sorted_controls)
            distances_norm = self.get_distance_probs(parent1, sorted_controls)
            id2, parent2 = self.parent_selection(distances_norm, sorted_controls)

            # create child and add to children list
            child = self.crossover_and_mutation(parent1, parent2, id1, id2, fitnesses)
            children.append(child)

        # select parents based on normilzed probabilites of the fitnesses who
        # do not survive
        sorted_controls = self.determine_survival(fit_norm_sorted, sorted_controls, children)

        return sorted_controls

class GA_distanceroulette_weightedNpoint_scramblemutation(GA_roulette_weightedNpoint_scramblemutation):
    """
    Roulette wheel selection with a heuristc based on the relative distance to 
    all parents, weighted N-point crossover and scramble mutiation
    """

    def get_distance_probs(self, parent1, sorted_controls):
        """
        Determines the  probability based on the distance of the first
        parent selected relatively to that of all the others.
        """
        distances, params1 = [], parent1.get_params()
        for other_parent in sorted_controls:
            params2 = other_parent.get_params()
            distance = 0
            
            for layer in range(self.tot_layers):
                weight_str = "W" + str(layer)
                W1, W2 = params1[weight_str], params2[weight_str]
                diff = (W1 - W2).flatten()
                distance += np.sqrt(np.dot(diff, diff))

            distances.append(distance)

        # normalize, so we get probabilities
        sum_distances = sum(distances)
        distances_norm = [distance / sum_distances for distance in distances]

        return distances_norm

    def make_new_generation(self, fit_norm_sorted, sorted_controls, fitnesses):
        """
        Crossover gense for a given population
        """

        # start creating children based on pairs of parents
        children = []
        for i in range(0, self.pop_size, self.nr_skip_parents):

            # roulette wheel selection of two parents and second parent selected
            # with help of the distance heuristic
            id1, parent1 = self.parent_selection(fit_norm_sorted, sorted_controls)
            distances_norm = self.get_distance_probs(parent1, sorted_controls)
            id2, parent2 = self.parent_selection(distances_norm, sorted_controls)

            # create child and add to children list
            child = self.crossover_and_mutation(parent1, parent2, id1, id2, fitnesses)
            children.append(child)

        # select parents based on normilzed probabilites of the fitnesses who
        # do not survive
        sorted_controls = self.determine_survival(fit_norm_sorted, sorted_controls, children)

        return sorted_controls

class GA_distanceroulette_weightedNpoint_adaptmutation(GA_roulette_weightedNpoint_adaptmutation):
    """
    Roulette wheel selection with a heuristc based on the relative distance to 
    all parents, weighted N-point crossover and self-adaptive mutiation
    """

    def get_distance_probs(self, parent1, sorted_controls):
        """
        Determines the  probability based on the distance of the first
        parent selected relatively to that of all the others.
        """
        distances, params1 = [], parent1.get_params()
        for other_parent in sorted_controls:
            params2 = other_parent.get_params()
            distance = 0
            
            for layer in range(self.tot_layers):
                weight_str = "W" + str(layer)
                W1, W2 = params1[weight_str], params2[weight_str]
                diff = (W1 - W2).flatten()
                distance += np.sqrt(np.dot(diff, diff))

            distances.append(distance)

        # normalize, so we get probabilities
        sum_distances = sum(distances)
        distances_norm = [distance / sum_distances for distance in distances]

        return distances_norm

    def make_new_generation(self, fit_norm_sorted, sorted_controls, fitnesses):
        """
        Crossover gense for a given population
        """

        # start creating children based on pairs of parents
        children = []
        for i in range(0, self.pop_size, self.nr_skip_parents):

            # roulette wheel selection of two parents and second parent selected
            # with help of the distance heuristic
            id1, parent1 = self.parent_selection(fit_norm_sorted, sorted_controls)
            distances_norm = self.get_distance_probs(parent1, sorted_controls)
            id2, parent2 = self.parent_selection(distances_norm, sorted_controls)

            # create child and add to children list
            child = self.crossover_and_mutation(parent1, parent2, id1, id2, fitnesses)
            children.append(child)

        # select parents based on normilzed probabilites of the fitnesses who
        # do not survive
        sorted_controls = self.determine_survival(fit_norm_sorted, sorted_controls, children)

        return sorted_controls

class GA_distanceroulette_weightedNpoint_adaptscramblemutation(GA_distanceroulette_weightedNpoint_adaptmutation):
    """
    Roulette wheel selection with a heuristc based on the relative distance to 
    all parents, weighted N-point crossover and self-adaptive mutiation combined
    with a scramble mutation
    """
    
    def mutation(self, child_params, str_layer, child_cont, parent1, parent2, W2, b2):
        """
        Performs self-adaptive mutation
        """
        
        # determine mutation step size that will be used from one of the parents
        if np.random.random() < 0.5:
            mutation_step_size = parent1.mutation_step_size
        else:
            mutation_step_size = parent2.mutation_step_size

        # get new mutation step size and set child's mutation step size equal to it
        mutation_step_size = mutation_step_size * np.exp(self.tau * np.random.normal(0, 1))
        child_cont.set_mutation_step_size(mutation_step_size)

        # add noise (self-adaptive mutation) for the weights
        for i in range(len(W2)):
            for j in range(len(W2[0])):
                if np.random.uniform(0, 1) < self.mutation_chance:
                    child_params["W" + str_layer][i][j] += mutation_step_size

        # add noise (self-adaptive mutation) for the biases
        for i in range(np.shape(b2)[0]):
            if np.random.uniform(0, 1) < self.mutation_chance:
                child_params["b" + str_layer][i] += mutation_step_size

        # scramble the child's parameters (scramble mutation)
        start_point = random.randint(1, 3)
        end_point = random.randint(start_point, len(child_params["W" + str_layer]))   
        for i in range(start_point, end_point):
            np.random.shuffle(child_params["W" + str_layer][i])                             
                
        return child_params