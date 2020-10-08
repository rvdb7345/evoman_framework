import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
import sys
from demo_controller import player_controller

sys.path.insert(0, 'evoman')

from environment import Environment
experiment_name = "best_solutions_test"
from scipy.stats import ttest_ind, levene, f_oneway


def play_game(pcont, enemies, multiplemode):
    """
    Helper function to simulate a game in the Evoman framework
    """
    env = Environment(
        experiment_name=experiment_name,
        enemies=enemies,
        multiplemode=multiplemode,
        playermode="ai",
        player_controller=player_controller(10),
        enemymode="static",
        level=2,
        speed="fastest",
        logs="off"
    )

    return env.play(pcont=pcont)


if __name__ == '__main__':

    ############################################## DGEA 7,8 ###########################################################
    df_DGEA_78_fitnesses = pd.read_csv('results/dgea_dgea_first_e_set/best_fits_best_solse7e8.csv')
    array_DGEA_78_sols = np.load('results/dgea_dgea_first_e_set/best_sols_e7e8.npz')

    sols_DGEA_78_array = [array_DGEA_78_sols['arr_{}'.format(i)] for i in range(len(array_DGEA_78_sols.keys()))]

    sorted_tuples_DGEA_78 = sorted(list(zip(df_DGEA_78_fitnesses.values, sols_DGEA_78_array)), key=lambda pair: pair[0])

    best_ten_DGEA_78 = sorted_tuples_DGEA_78[-10:]

    repetitions = 5

    controllers = np.array([tuple_fitcont[1] for tuple_fitcont in best_ten_DGEA_78])

    # create input including the number of neurons and the enemies so this isn't in the simulate function
    enemies = [1,2, 3,4,5,6,7,8]
    pool_input = [(pcont, [enemy], "no") for pcont in controllers for i in range(repetitions) for enemy in enemies]

    # run the simulations in parallel
    pool = Pool(24)
    pool_list = pool.starmap(play_game, pool_input)
    pool.close()
    pool.join()

    # get the fitnesses from the total results formatted as [(f, p, e, t), (...), ...]
    fitnesses = [pool_list[i][0] for i in range(len(pool_list))]
    player_lifes = [pool_list[i][1] for i in range(len(pool_list))]
    enemies_lifes = [pool_list[i][2] for i in range(len(pool_list))]

    column_per_enemy_fitness = np.reshape(fitnesses, (int(len(pool_list) / len(enemies)), len(enemies)))
    mean_fitness_per_enemy = np.mean(column_per_enemy_fitness, axis=0)
    max_fitness_per_enemy = np.amax(column_per_enemy_fitness, axis=0)
    print("mean fitness per enemy", mean_fitness_per_enemy)
    print("max fitness per enemy: ", max_fitness_per_enemy)

    column_per_enemy_ind_gain = np.reshape(player_lifes, (int(len(pool_list) / len(enemies)), len(enemies))) - np.reshape(enemies_lifes, (int(len(pool_list) / len(enemies)), len(enemies)))
    mean_indgain_per_enemy = np.mean(column_per_enemy_ind_gain, axis=0)
    max_indgain_per_enemy = np.amax(column_per_enemy_ind_gain, axis=0)
    print("mean indgain per enemy", mean_indgain_per_enemy)
    print("max indgain per enemy: ", max_indgain_per_enemy)

    print(column_per_enemy_ind_gain)

    max_indgain_per_enemy = np.sum(column_per_enemy_ind_gain, axis=1)
    print('summed individual gain: ', max_indgain_per_enemy)

    ############################################## DGEA 2,6 ###########################################################
    # df_DGEA_78_fitnesses = pd.read_csv('results/dgea_dgea_second_e_set/best_fits_best_solse2e6.csv')
    # array_DGEA_78_sols = np.load('results/dgea_dgea_second_e_set/best_sols_e2e6.npz')
    #
    # sols_DGEA_78_array = [array_DGEA_78_sols['arr_{}'.format(i)] for i in range(len(array_DGEA_78_sols.keys()))]
    #
    # sorted_tuples_DGEA_78 = sorted(list(zip(df_DGEA_78_fitnesses.values, sols_DGEA_78_array)), key=lambda pair: pair[0])
    #
    # best_ten_DGEA_78 = sorted_tuples_DGEA_78[-10:]
    #
    # repetitions = 5
    #
    # controllers = np.array([tuple_fitcont[1] for tuple_fitcont in best_ten_DGEA_78])
    #
    # # create input including the number of neurons and the enemies so this isn't in the simulate function
    # enemies = [1,2,3,4,5,6,7,8]
    # pool_input = [(pcont, [enemy], "no") for pcont in controllers for i in range(repetitions) for enemy in enemies]
    #
    # # run the simulations in parallel
    # pool = Pool(24)
    # pool_list = pool.starmap(play_game, pool_input)
    # pool.close()
    # pool.join()
    #
    # # get the fitnesses from the total results formatted as [(f, p, e, t), (...), ...]
    # fitnesses = [pool_list[i][0] for i in range(len(pool_list))]
    # player_lifes = [pool_list[i][1] for i in range(len(pool_list))]
    # enemies_lifes = [pool_list[i][2] for i in range(len(pool_list))]
    #
    # column_per_enemy_fitness = np.reshape(fitnesses, (int(len(pool_list) / len(enemies)), len(enemies)))
    # mean_fitness_per_enemy = np.mean(column_per_enemy_fitness, axis=0)
    # max_fitness_per_enemy = np.amax(column_per_enemy_fitness, axis=0)
    # print("mean fitness per enemy", mean_fitness_per_enemy)
    # print("max fitness per enemy: ", max_fitness_per_enemy)
    #
    # column_per_enemy_ind_gain = np.reshape(player_lifes, (int(len(pool_list) / len(enemies)), len(enemies))) - np.reshape(enemies_lifes, (int(len(pool_list) / len(enemies)), len(enemies)))
    # print(column_per_enemy_ind_gain)
    # mean_indgain_per_enemy = np.mean(column_per_enemy_ind_gain, axis=0)
    # max_indgain_per_enemy = np.amax(column_per_enemy_ind_gain, axis=0)
    # print("mean indgain per enemy", mean_indgain_per_enemy)
    # print("max indgain per enemy: ", max_indgain_per_enemy)
    #
    # max_indgain_per_enemy = np.sum(column_per_enemy_ind_gain, axis=1)
    # print('summed individual gain: ', max_indgain_per_enemy)



    # ############################################## NB 7,8 ###########################################################
    # df_DGEA_78_fitnesses = pd.read_csv('results/dgea_newblood_second_e_set/best_fits_best_solse2e6.csv')
    # array_DGEA_78_sols = np.load('results/dgea_newblood_second_e_set/best_sols_e2e6.npz')
    #
    # sols_DGEA_78_array = [array_DGEA_78_sols['arr_{}'.format(i)] for i in range(len(array_DGEA_78_sols.keys()))]
    #
    # sorted_tuples_DGEA_78 = sorted(list(zip(df_DGEA_78_fitnesses.values, sols_DGEA_78_array)), key=lambda pair: pair[0])
    #
    # best_ten_DGEA_78 = sorted_tuples_DGEA_78[-10:]
    #
    # repetitions = 5
    #
    # controllers = np.array([tuple_fitcont[1] for tuple_fitcont in best_ten_DGEA_78])
    #
    # # create input including the number of neurons and the enemies so this isn't in the simulate function
    # enemies = [1,2, 3,4,5,6,7,8]
    # pool_input = [(pcont, [enemy], "no") for pcont in controllers for i in range(repetitions) for enemy in enemies]
    #
    # # run the simulations in parallel
    # pool = Pool(24)
    # pool_list = pool.starmap(play_game, pool_input)
    # pool.close()
    # pool.join()
    #
    # # get the fitnesses from the total results formatted as [(f, p, e, t), (...), ...]
    # fitnesses = [pool_list[i][0] for i in range(len(pool_list))]
    # player_lifes = [pool_list[i][1] for i in range(len(pool_list))]
    # enemies_lifes = [pool_list[i][2] for i in range(len(pool_list))]
    #
    # column_per_enemy_fitness = np.reshape(fitnesses, (int(len(pool_list) / len(enemies)), len(enemies)))
    # mean_fitness_per_enemy = np.mean(column_per_enemy_fitness, axis=0)
    # max_fitness_per_enemy = np.amax(column_per_enemy_fitness, axis=0)
    # print("mean fitness per enemy", mean_fitness_per_enemy)
    # print("max fitness per enemy: ", max_fitness_per_enemy)
    #
    # column_per_enemy_ind_gain = np.reshape(player_lifes, (int(len(pool_list) / len(enemies)), len(enemies))) - np.reshape(enemies_lifes, (int(len(pool_list) / len(enemies)), len(enemies)))
    # mean_indgain_per_enemy = np.mean(column_per_enemy_ind_gain, axis=0)
    # max_indgain_per_enemy = np.amax(column_per_enemy_ind_gain, axis=0)
    # print("mean indgain per enemy", mean_indgain_per_enemy)
    # print("max indgain per enemy: ", max_indgain_per_enemy)



