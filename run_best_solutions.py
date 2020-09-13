import pickle
import sys, os

sys.path.insert(0, "evoman")

from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from environment import Environment
from controller_julien import test_controller
from controller import Controller
import numpy as np

experiment_name = "best_solutions_test"
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


if __name__ == '__main__':
    with open('best_results', 'rb') as config_df_file:
        config_df = pickle.load(config_df_file)

    enemies = 8
    repetitions = 5
    selected_df = config_df.loc[config_df['enemies'] == enemies]

    controllers = selected_df['best_solution'].values

    # create input including the number of neurons and the enemies so this isn't in the simulate function
    pool_input = [(pcont, [enemies], "no") for pcont in controllers for i in range(repetitions)]

    # run the simulations in parallel
    pool = Pool(cpu_count())
    pool_list = pool.starmap(play_game, pool_input)
    pool.close()
    pool.join()

    # pool_list = []
    # for i in range(len(pool_input)):
    #     pool_list.append(play_game(pool_input[i][0], pool_input[i][1], pool_input[i][2]))

    # get the fitnesses from the total results formatted as [(f, p, e, t), (...), ...]
    fitnesses = [pool_list[i][0] for i in range(len(pool_list))]
    player_lifes = [pool_list[i][1] for i in range(len(pool_list))]
    enemies_lifes = [pool_list[i][2] for i in range(len(pool_list))]

    fitnesses_row_per_sol = np.reshape(fitnesses, (int(len(pool_list) / repetitions), repetitions))
    player_lifes_row_per_sol = np.reshape(player_lifes, (int(len(pool_list) / repetitions), repetitions))
    enemies_lifes_row_per_sol = np.reshape(enemies_lifes, (int(len(pool_list) / repetitions), repetitions))

    individual_gain = player_lifes_row_per_sol - enemies_lifes_row_per_sol
    mean_individual_gain = np.mean(individual_gain, axis=1)

    plt.figure()
    plt.boxplot(mean_individual_gain)
    plt.show()
