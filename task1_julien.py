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
    Cross genes for each layer in hidden network according to a relative
    probability based on the fitnes of each parent
    """
    child_cont = test_controller(nn_topology)
    child_params, n_layers = {}, len(nn_topology)
    network1, network2 = parent1.get_params(), parent2.get_params()

    # assert n_layers == len(network1) and n_layers == len(network2), "ERROR"

    for layer in range(n_layers):
        str_layer = str(layer)
        W1, W2 = network1["W" + str_layer], network2["W" + str_layer]
        b1, b2 = network1["b" + str_layer], network2["b" + str_layer]
        activation_funcs = network1["activation" + str_layer], network2["activation" + str_layer]
        child_params["W" + str_layer] = prob1 * W1 + prob2 * W2
        child_params["b" + str_layer] = prob1 * b1 + prob2 * b2

        # determine activation function by chance
        active_func = np.random.choice(activation_funcs, p=[prob1, prob2])
        child_params["activation" + str_layer] = active_func

        # add noise (mutation)
        if np.random.uniform(0, 1) < mutation_chance:
            noise = np.random.normal(0, 1)
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

def roulette_wheel_selection(fit_norm, pcontrols, id_prev=-1):
    """
    """

    random_number = np.random.uniform(0, 1)
    for idx, norm in enumerate(fit_norm):
        if random_number < norm and not id_prev != idx:
            return idx, pcontrols[idx]
        elif random_number < norm and idx + 1 < len(pcontrols):
            idx, pcontrols[idx + 1]

    return idx, pcontrols[len(pcontrols) - 1]

def make_new_generation(pop_size, int_skip, nn_topology, fitnesses, sorted_controls, mutation_chance):
    """
    Crossover gense for a given population
    """

    # start creating childrens by pairs (for only a quarter of the population)
    children = []
    for i in range(0, pop_size, int_skip):

        # # select parents and determine the relative probability of retrieving
        # # the relative amount of genes from each one of the parents
        # id1 = np.random.randint(0, pop_size)
        # id2 = np.random.randint(0, pop_size)
        # parent1, parent2 = pcontrols[id1], pcontrols[id2]

        # prob1, prob2 = 1, 0
        # if fitnesses[id1] != fitnesses[id2]:
        #     fitness1 = np.abs(fitnesses[id1])
        #     fitness2 = np.abs(fitnesses[id2])
        #     prob1 = fitness1 / (fitness1 + fitness2)
        #     prob2 = 1 - prob1

        # select parents with roullete wheel selection
        id1, parent1 = roulette_wheel_selection(fitnesses, sorted_controls)
        id2, parent2 = roulette_wheel_selection(fitnesses, sorted_controls, id_prev=id1)
        prob1 = np.random.uniform(0, 1)
        prob2 = 1 - prob1

        # create child and add to children list
        child = crossover_and_mutation(parent1, parent2, prob1, prob2, nn_topology, mutation_chance)
        children.append(child)

    # replace the parents with the lowest score with the newly made children
    # and update population
    sorted_controls[0:len(children)] = children

    return sorted_controls

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
    # num_cores = cpu_count()

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

    # create player controls (random neural network) for the entire population
    # pcontrols = [test_controller(nn_topology) for _ in range(population_size)]
    pcontrols = []
    for _ in range(population_size):
        pcont = test_controller(nn_topology)
        pcont.initialize_random_network()
        pcontrols.append(pcont)

    # result = play_game(pcontrols[0], enemies, "no")
    # print(result)

    fitnesses = np.zeros(population_size)
    mean_fitness_gens = np.zeros(n_generations + 1)
    stds_fitness_gens = np.zeros(n_generations + 1)

    best_fit, best_sol, not_improved = 0, None, 0

    for gen in tqdm(range(n_generations)):

        # create input including the number of neurons and the enemies so this isn't in the simulate function
        pool_input = [(pcont, enemies, "no") for pcont in pcontrols]

        # run the simulations in parallel
        pool = Pool(cpu_count())
        pool_list = pool.starmap(play_game, pool_input)
        pool.close()
        pool.join()

        # get the fitnesses from the total results formatted as [(f, p, e, t), (...), ...]
        fitnesses = [pool_list[i][0] for i in range(population_size)]

        # create a (proportional) cdf for the fitnesses
        sorted_controls = [
        parent for _, parent in sorted(
                                    list(zip(fitnesses, pcontrols)),
                                    key=lambda x: x[0])
        ]
        fitnesses.sort()
        best_fit_gen, worst_fit_gen = max(fitnesses), min(fitnesses)
        fit_norm = [(fit - worst_fit_gen) / (best_fit_gen - worst_fit_gen) for fit in fitnesses]

        print("Fitnesses Gen {} are {}".format(gen, fitnesses))

        # # normalize fitness and determine probabilites based on fitness
        # best_fit_gen, worst_fit_gen = max(fitnesses), min(fitnesses)
        # fit_norm = [(fit - worst_fit_gen) / (best_fit_gen - worst_fit_gen) for fit in fitnesses]
        # probs = fit_norm

        best_fit_gen = max(fitnesses)
        if best_fit_gen >= best_fit:
            best_fit = best_fit_gen
            best_sol = pcontrols[fitnesses.index(best_fit)]
            not_improved = 0
        else:
            not_improved += 1

        mean_fitness_gens[gen] = np.mean(fitnesses)
        stds_fitness_gens[gen] = np.std(fitnesses)
        print(
            "Generation: {}, with an average fitness: {} and standard deviation: {}"
            .format(gen, mean_fitness_gens[gen], stds_fitness_gens[gen])
        )

        # make new generation
        pcontrols = make_new_generation(
            population_size, int_skip, nn_topology, fit_norm, sorted_controls, mutation_chance
        )
        # pcontrols = make_new_generation(population_size, 2, nn_topology, fitnesses, pcontrols)

    # run final solution in parallel
    fitnesses = []
    pool_input = [(pcont, enemies, "no") for pcont in pcontrols]
    pool = Pool(cpu_count())
    pool_list = pool.starmap(play_game, pool_input)
    pool.close()
    pool.join()

    # get the fitnesses from the total results formatted as [(f, p, e, t), (...), ...]
    fitnesses = [pool_list[i][0] for i in range(population_size)]

    # check if it is a better solution, if so then update
    best_fit_gen = max(fitnesses)
    if best_fit_gen >= best_fit:
        best_fit = best_fit_gen
        best_sol = pcontrols[fitnesses.index(best_fit)]

    # save final result
    mean_fitness_gens[n_generations] = np.mean(fitnesses)
    stds_fitness_gens[n_generations] = np.std(fitnesses)

    print('Final population solution has an average fitness of: {}'.format(np.mean(fitnesses)))
    print("Best fit found: {}".format(best_fit))

    # plot the results (mean and standard deviation) over the generations
    plt.figure()
    plt.title("Fitness per generation")
    plt.errorbar(
        np.arange(0, n_generations + 1), mean_fitness_gens, yerr=stds_fitness_gens
    )
    plt.grid()
    plt.xlabel("Generation (#)")
    plt.ylabel("Fitness")
    plt.show()

    # # # initializes environment with ai player using random controller, playing against static enemy
    # env = Environment(
    #     experiment_name=experiment_name, playermode="ai",
    #     player_controller=pcont,
    #     sound="off"
    # )
    # x = env.play(pcont=pcont)
    # print(type(x))
