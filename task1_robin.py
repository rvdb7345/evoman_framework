################################
# EvoMan FrameWork 2020        #
# Author: Julien Fer           #
# julienrmfer@gmail.com        #
################################

# imports framework
import sys, os
sys.path.insert(0, "evoman")

from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
from environment import Environment
from controller_julien import test_controller

dir_path = os.path.abspath('')

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

def N_crossover_and_adaptive_mutation(parent1, parent2, prob1, prob2, nn_topology, mutation_chance, tau, lb=-1, ub=1):
    """
    Cross genes for each layer in hidden network according to a relative
    probability based on the fitnes of each parent
    """
    child_cont = test_controller(nn_topology)
    child_params, n_layers = {}, len(nn_topology)
    network1, network2 = parent1.get_params(), parent2.get_params()

    for layer in range(n_layers):
        str_layer = str(layer)
        W1, W2 = network1["W" + str_layer], network2["W" + str_layer]
        b1, b2 = network1["b" + str_layer], network2["b" + str_layer]

        activation_funcs = network1["activation" + str_layer], network2["activation" + str_layer]

        # print("Prob 1:", prob1)
        child_params["W" + str_layer] = np.array([[W1[i][j] if np.random.random() < prob1 else W2[i][j]
                                                   for i in range(len(W2))] for j in range(len(W1[0]))]).T
        child_params["b" + str_layer] = np.array([b1[i] if np.random.random() < prob1 else b2[i]
                                                  for i in range(np.shape(b2)[0])])

        # determine activation function by chance
        active_func = np.random.choice(activation_funcs, p=[prob1, prob2])
        child_params["activation" + str_layer] = active_func

        if np.random.random() < 0.5:
            mutation_step_size = parent1.mutation_step_size
        else:
            mutation_step_size = parent2.mutation_step_size

        mutation_step_size = mutation_step_size * np.exp(tau * np.random.normal(0, 1))

        child_cont.set_mutation_step_size(mutation_step_size)

        # add noise (mutation)
        for i in range(len(W2)):
            for j in range(len(W2[0])):
                if np.random.uniform(0, 1) < mutation_chance:
                    child_params["W" + str_layer][i][j] += mutation_step_size


        for i in range(np.shape(b2)[0]):
            child_params["b" + str_layer][i] += mutation_step_size

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


def make_new_generation(pop_size, int_skip, nn_topology, fitnesses, sorted_controls, mutation_chance, tau):
    """
    Crossover gense for a given population
    """

    # start creating childrens by pairs (for only a quarter of the population)
    # truncated "killing" selection
    children = []
    for i in range(0, pop_size, int_skip):
        # select parents with roullete wheel selection
        id1, parent1 = roulette_wheel_selection(fitnesses, sorted_controls)
        id2, parent2 = roulette_wheel_selection(fitnesses, sorted_controls, id_prev=id1)
        prob1 = fitnesses[id1] / (fitnesses[id2] + fitnesses[id1])
        prob2 = 1 - prob1

        # create child and add to children list
        child = N_crossover_and_adaptive_mutation(parent1, parent2, prob1, prob2, nn_topology, mutation_chance, tau)
        children.append(child)

    # replace the parents with the lowest score with the newly made children
    # and update population (this can also be changed)
    sorted_controls[0:len(children)] = children

    return sorted_controls

def run_one_parallel(
    pcontrols, enemies, pop_size, best_fit, gen,
    not_improved, mean_fitness_gens, stds_fitness_gens,
        mean_p_lifes_gens, stds_p_lifes_gens,
        mean_e_lifes_gens, stds_e_lifes_gens,
        best_sol
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
    player_lifes = [pool_list[i][1] for i in range(population_size)]
    enemies_lifes = [pool_list[i][2] for i in range(population_size)]

    best_fit_gen = max(fitnesses)
    if best_fit_gen > best_fit:
        best_fit = best_fit_gen
        best_sol = pcontrols[fitnesses.index(best_fit)]
        not_improved = 0
    else:
        not_improved += 1

    mean_fitness_gens[gen] = np.mean(fitnesses)
    stds_fitness_gens[gen] = np.std(fitnesses)

    mean_p_lifes_gens[gen] = np.mean(player_lifes)
    stds_p_lifes_gens[gen] = np.std(player_lifes)

    mean_e_lifes_gens[gen] = np.mean(enemies_lifes)
    stds_e_lifes_gens[gen] = np.std(enemies_lifes)

    return fitnesses, best_fit, player_lifes, enemies, best_sol


def save_generations(generation_sum_df):
    if os.path.exists(os.path.join(dir_path, 'generational_summary')):
        with open(os.path.join(dir_path, 'generational_summary'), 'rb') as config_df_file:
            config_df = pickle.load(config_df_file)
            generation_sum_df = pd.concat([generation_sum_df, config_df])

    with open('generational_summary', 'wb') as config_dictionary_file:
        pickle.dump(generation_sum_df, config_dictionary_file)


def save_best_solution(enemies, best_fit, sol):
    best_solution_df = pd.DataFrame({'enemies': enemies, 'fitness': best_fit, 'best_solution': sol})

    if os.path.exists(os.path.join(dir_path, 'best_results')):
        with open(os.path.join(dir_path, 'best_results'), 'rb') as config_df_file:
            config_df = pickle.load(config_df_file)
            best_solution_df = pd.concat([best_solution_df, config_df])

    with open('best_results', 'wb') as config_dictionary_file:
        pickle.dump(best_solution_df, config_dictionary_file)



def simulate_parallel(
        nn_topology, pop_size, n_gens,
        lu=-1, ub=1, int_skip=4,
        mutation_prob=0.2, enemies=[8], multiplemode="no"
    ):

    # create player controls (random neural network) for the entire population
    # pcontrols = [test_controller(nn_topology) for _ in range(population_size)]

    tau = 1 / pop_size ** 2

    pcontrols = []
    for _ in range(pop_size):
        pcont = test_controller(nn_topology)
        pcont.initialize_random_network()
        pcontrols.append(pcont)

    # fitnesses = np.zeros(pop_size)
    mean_fitness_gens = np.zeros(n_gens + 1)
    stds_fitness_gens = np.zeros(n_gens + 1)

    mean_p_lifes_gens = np.zeros(n_generations + 1)
    stds_p_lifes_gens = np.zeros(n_generations + 1)

    mean_e_lifes_gens = np.zeros(n_generations + 1)
    stds_e_lifes_gens = np.zeros(n_generations + 1)

    best_fit, best_sol, not_improved = 0, None, 0

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

    generation_sum_df = pd.DataFrame(columns=['datetime', 'gen', 'enemies', 'fit_max', 'fit_mean'])

    # start evolutionary algorithm
    for gen in tqdm(range(n_gens)):

        fitnesses, best_fit, player_lifes, enemies_lifes, best_sol = run_one_parallel(
            pcontrols, enemies, pop_size, best_fit, gen,
            not_improved, mean_fitness_gens, stds_fitness_gens,
            mean_p_lifes_gens, stds_p_lifes_gens, mean_e_lifes_gens, stds_e_lifes_gens,
            best_sol
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

        generation_sum_df = generation_sum_df.append({'datetime': dt_string, 'gen': gen, 'enemies': enemies[0], 'fit_max': max(fitnesses),
                                      'fit_mean': mean_fitness_gens[gen]}, ignore_index=True)

        # make new generation
        pcontrols = make_new_generation(
            population_size, int_skip, nn_topology, fit_norm, sorted_controls, mutation_prob, tau
        )

    # run final solution in parallel
    fitnesses, best_fit, player_lifes, enemies_lifes, best_sol = run_one_parallel(
            pcontrols, enemies, pop_size, best_fit, n_gens,
            not_improved, mean_fitness_gens, stds_fitness_gens,
            mean_p_lifes_gens, stds_p_lifes_gens, mean_e_lifes_gens, stds_e_lifes_gens,
            best_sol
    )

    print(generation_sum_df)

    # save best solution
    save_best_solution(enemies, best_fit, best_sol)

    # save the mean and the max fitness during each run
    save_generations(generation_sum_df)

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

    # plot the results (mean and standard deviation) over the generations
    plt.figure()
    plt.title("Player lifes per generation")
    plt.errorbar(
        np.arange(0, n_generations + 1), mean_p_lifes_gens, yerr=stds_p_lifes_gens
    )
    plt.grid()
    plt.xlabel("Generation (#)")
    plt.ylabel("Life")
    plt.show()

    # plot the results (mean and standard deviation) over the generations
    plt.figure()
    plt.title("Enemy lifes per generation")
    plt.errorbar(
        np.arange(0, n_generations + 1), mean_e_lifes_gens, yerr=stds_e_lifes_gens
    )
    plt.grid()
    plt.xlabel("Generation (#)")
    plt.ylabel("Life")
    plt.show()


if __name__ == "__main__":

    # set the parameters
    inputs, n_hidden_neurons, outputs = 20, 10, 5
    enemies = [8]
    lower_bound = -1
    upper_bound = 1
    population_size = 100
    n_generations = 25
    mutation_chance = 0.2
    int_skip = 4
    tau = 1 / population_size**2

    repeats = 8
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
    for i in range(repeats):
        simulate_parallel(
            nn_topology, population_size, n_generations,
            lu=lower_bound, ub=upper_bound, int_skip=int_skip,
            mutation_prob=mutation_chance, enemies=enemies, multiplemode="no"
        )

    # # # initializes environment with ai player using random controller, playing against static enemy
    # env = Environment(
    #     experiment_name=experiment_name, playermode="ai",
    #     player_controller=pcont,
    #     sound="off"
    # )
    # x = env.play(pcont=pcont)
    # print(type(x))
