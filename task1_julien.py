################################
# EvoMan FrameWork 2020        #
# Author: Julien Fer           #
# julienrmfer@gmail.com        #
################################

# imports framework
import sys, os
sys.path.insert(0, "evoman")

from multiprocessing import Pool, cpu_count
from tqdm  import tqdm
import numpy as np
import matplotlib.pyplot as plt
from environment import Environment
from controller_julien import test_controller

experiment_name = "task1_julien"
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

def play_game(pcont, enemies, multiplemode):
    """
    Helper function to simulate a game in the Evoman framework
    """
    env = Environment(
        experiment_name=experiment_name,
        enemies=enemies,
        multiplemode=multiplemode,
        playermode="ai",
        player_controller=pcont,
        enemymode="static",
        level=2,
        speed="fastest",
        logs="off"
    )

    return env.play(pcont=pcont)

def crossover_and_mutation(parent1, parent2, prob1, prob2, nn_topology, mutation_chance, lb=-1, ub=1):
    """
    Cross genes for each layer in hidden network by weighted linear combination.
    The probability is for now drawn from a uniform distribution (see make_new_generation)

    (according to a relative probability based on the fitnes of each parent??? --> 
    less diversity, cause a greedy approach --> so maybe not necessary?)
    """

    # setup variables for child controller and parametrs, nr of layer and
    # parameters of parents
    child_cont = test_controller(nn_topology)
    child_params, n_layers = {}, len(nn_topology)
    network1, network2 = parent1.get_params(), parent2.get_params()

    # performs crossover per layer 
    for layer in range(n_layers):
        str_layer = str(layer)

        # retrieve matrices parents and perform weighted linear combination
        W1, W2 = network1["W" + str_layer], network2["W" + str_layer]
        b1, b2 = network1["b" + str_layer], network2["b" + str_layer]
        activation_funcs = network1["activation" + str_layer], network2["activation" + str_layer]
        child_params["W" + str_layer] = prob1 * W1 + prob2 * W2
        child_params["b" + str_layer] = prob1 * b1 + prob2 * b2

        # determine activation function by same probabilities
        active_func = np.random.choice(activation_funcs, p=[prob1, prob2])
        child_params["activation" + str_layer] = active_func

        # add noise (mutation)
        if np.random.uniform(0, 1) < mutation_chance:
            noise = np.random.normal(0, 1) # this can be changed (uniform distribution)
            child_params["W" + str_layer] += noise
            child_params["b" + str_layer] += noise

        # adjust for limits weights
        weights_child = child_params["W" + str_layer]
        bias_child = child_params["b" + str_layer]
        weights_child[weights_child > ub] = ub
        weights_child[weights_child < lb] = lb
        bias_child[bias_child > ub] = ub
        bias_child[bias_child < lb] = lb

    child_cont.create_network(child_params)

    return child_cont

def roulette_rank_selection(fit_norm, sorted_controls):
    """
    """
    return np.random.choice(sorted_controls, p=fit_norm)

def roulette_wheel_selection(fit_norm, pcontrols, id_prev=-1):
    """
    Selects parent controller by means of roulette wheel selection.
    Note, that it asssumes that the fitness is normalized between 0 and 1.

    ROULETTE WHEEL IS NOT THE BASED OPTION WHEN VALUES CAN BE NEGATIVE
    --> possible solution is to firs rank the parents and determine the probability
        on their ranking.
    """

    # checks !!!! THIS NEEDS TO BE CORRECTED 
    random_number, prob = np.random.uniform(0.0, 1.0), 0.0
    for idx, norm in enumerate(fit_norm):
        prob += norm
        if random_number < prob and not id_prev != idx:
            return idx, pcontrols[idx]
        elif random_number < prob and idx + 1 < len(pcontrols):
            return idx, pcontrols[idx + 1]

    return idx, pcontrols[len(pcontrols) - 1]

def make_new_generation(pop_size, int_skip, nn_topology, fitnesses, sorted_controls, mutation_chance):
    """
    Crossover gense for a given population
    """

    # start creating childrens by pairs (for only a quarter of the population)
    # truncated "killing" selection
    children = []
    for i in range(0, pop_size, int_skip):

        ## RANK SELECTION
        parent1 = roulette_rank_selection(fitnesses, sorted_controls)
        parent2 = roulette_rank_selection(fitnesses, sorted_controls)
        prob1 = np.random.uniform(0, 1)
        prob2 = 1 - prob1

        # # select parents with roullete wheel selection
        # id1, parent1 = roulette_wheel_selection(fitnesses, sorted_controls)
        # id2, parent2 = roulette_wheel_selection(fitnesses, sorted_controls)
        # prob1 = np.random.uniform(0, 1)
        # prob2 = 1 - prob1

        # create child and add to children list
        child = crossover_and_mutation(parent1, parent2, prob1, prob2, nn_topology, mutation_chance)
        children.append(child)

    # replace the parents with the lowest score with the newly made children
    # and update population (this can also be changed)
    sorted_controls[0:len(children)] = children

    return sorted_controls

def run_one_parallel(
    pcontrols, enemies, pop_size, best_fit, gen, 
    not_improved, mean_fitness_gens, stds_fitness_gens
    ):
    """
    Runs one parralel simulation in the Evoman framework
    """

    # create input including the number of neurons and the enemies so this isn't in the simulate function
    pool_input = [(pcont, enemies, "no") for pcont in pcontrols]

    # run the simulations in parallel
    pool = Pool(cpu_count())
    pool_list = pool.starmap(play_game, pool_input)
    pool.close()
    pool.join()

    # get the fitnesses from the total results formatted as [(f, p, e, t), (...), ...]
    fitnesses = [pool_list[i][0] for i in range(pop_size)]

    best_fit_gen = max(fitnesses)
    if best_fit_gen > best_fit:
        best_fit = best_fit_gen
        best_sol = pcontrols[fitnesses.index(best_fit)]
        not_improved = 0
    else:
        not_improved += 1

    mean_fitness_gens[gen] = np.mean(fitnesses)
    stds_fitness_gens[gen] = np.std(fitnesses)

    return fitnesses, best_fit

def simulate_parallel(
        nn_topology, pop_size, n_gens, 
        lu=-1, ub=1, int_skip=4, 
        mutation_prob=0.2, enemies=[8], multiplemode="no"
    ):

    # create player controls (random neural network) for the entire population
    # pcontrols = [test_controller(nn_topology) for _ in range(population_size)]
    pcontrols = []
    for _ in range(pop_size):
        pcont = test_controller(nn_topology)
        pcont.initialize_random_network()
        pcontrols.append(pcont)

    # fitnesses = np.zeros(pop_size)
    mean_fitness_gens = np.zeros(n_gens + 1)
    stds_fitness_gens = np.zeros(n_gens + 1)
    best_fit, best_sol, not_improved = 0, None, 0

    # start evolutionary algorithm
    for gen in tqdm(range(n_gens)):

        fitnesses, best_fit = run_one_parallel(
            pcontrols, enemies, pop_size, best_fit, gen, 
            not_improved, mean_fitness_gens, stds_fitness_gens
        )
        print("Best fit is", best_fit)

        # create a (proportional) cdf for the fitnesses
        sorted_controls = [
            parent for _, parent in sorted(
                                        list(zip(fitnesses, pcontrols)),
                                        key=lambda x: x[0]
                                    )
        ]
        fitnesses.sort()

        ## !!!!! THIS IS FOR RANK SELECTION
        ranks = list(range(1, pop_size + 1, 1))
        sum_ranks = sum(ranks)
        fit_norm = [rank / sum_ranks for rank in ranks]

        # # CHECK FOR WHEN MIN AND MAX ARE EQUAL!!!!! --> Random Selection (or see note roulette wheel)
        # best_fit_gen, worst_fit_gen = max(fitnesses), min(fitnesses)
        # fit_norm = []
        # if best_fit_gen  != worst_fit_gen:
        #     fit_norm = [(fit - worst_fit_gen) / (best_fit_gen - worst_fit_gen) for fit in fitnesses]
        # else:
        #     fit_norm = [1 / pop_size] * pop_size
        # # print("Normalized fitness is", fit_norm)

        print(
            "Generation: {}, with an average fitness: {} and standard deviation: {}"
            .format(gen, round(mean_fitness_gens[gen], 2), round(stds_fitness_gens[gen], 2))
        )

        # make new generation
        pcontrols = make_new_generation(
            pop_size, int_skip, nn_topology, fit_norm, sorted_controls, mutation_prob
        )

    # run final solution in parallel
    fitnesses, best_fit = run_one_parallel(
            pcontrols, enemies, pop_size, best_fit, n_gens, 
            not_improved, mean_fitness_gens, stds_fitness_gens
    )

    print('Final population solution has an average fitness of: {}'.format(
            round(mean_fitness_gens[n_gens], 2)
        )
    )
    print("Best fit found: {}".format(best_fit))

    # plot the results (mean and standard deviation) over the generations
    plt.figure()
    plt.title("Fitness per generation")
    plt.errorbar(
        np.arange(0, n_gens + 1), mean_fitness_gens, yerr=stds_fitness_gens
    )
    plt.grid()
    plt.xlabel("Generation (#)")
    plt.ylabel("Fitness")
    plt.show()

if __name__ == "__main__":

    # set the parameters
    inputs, n_hidden_neurons, outputs = 20, 10, 5
    enemies = [8]
    lower_bound = -1
    upper_bound = 1
    population_size = 10
    n_generations = 5
    mutation_chance = 0.2
    int_skip = 4

    # this if for one hidden layer neural network
    nn_topology = [
        {"input_dim": inputs, "output_dim": n_hidden_neurons, "activation": "sigmoid"},
        {"input_dim": n_hidden_neurons, "output_dim": outputs, "activation": "sigmoid"}
    ]

    # # this is two layers
    # nn_topology = [
    #     {"input_dim": inputs, "output_dim": n_hidden_neurons, "activation": "sigmoid"},
    #     {"input_dim": n_hidden_neurons, "output_dim": n_hidden_neurons, "activation": "sigmoid"},
    #     {"input_dim": n_hidden_neurons, "output_dim": outputs, "activation": "sigmoid"}
    # ]

    simulate_parallel(
        nn_topology, population_size, n_generations, 
        lu=lower_bound, ub=upper_bound, int_skip=int_skip, 
        mutation_prob=mutation_chance, enemies=enemies, multiplemode="no"
    )

    # initializes environment with ai player using random controller, playing against static enemy
    # pcont = test_controller(nn_topology)
    # pcont.initialize_random_network()
    # env = Environment(
    #     experiment_name=experiment_name, playermode="ai",
    #     player_controller=pcont,
    #     sound="off"
    # )
    # x = env.play(pcont=pcont)
    # print(type(x))
