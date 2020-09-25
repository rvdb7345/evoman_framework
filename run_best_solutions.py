import pickle
import sys, os

sys.path.insert(0, "evoman")

from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from environment import Environment
from controller_julien import test_controller
from controller import Controller
import numpy as np
from scipy.stats import ttest_ind, levene, f_oneway

experiment_name = "best_solutions_test"
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


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

    ### --------------------------------- results 1 --------------------------------------- ###


    enemies = 8
    model = 'roulette_weightedNpoint_adaptmutation'
    name = 'robin'


    with open('best_results_1', 'rb') as config_df_file:
        config_df = pickle.load(config_df_file)


    repetitions = 5
    selected_df = config_df.loc[(config_df['enemies'] == enemies) & (config_df['model'] == model)]

    controllers = selected_df['best_solution'].values

    # create input including the number of neurons and the enemies so this isn't in the simulate function
    pool_input = [(pcont, [enemies], "no") for pcont in controllers for i in range(repetitions)]

    # run the simulations in parallel
    pool = Pool(cpu_count())
    pool_list = pool.starmap(play_game, pool_input)
    pool.close()
    pool.join()

    # get the fitnesses from the total results formatted as [(f, p, e, t), (...), ...]
    fitnesses = [pool_list[i][0] for i in range(len(pool_list))]
    player_lifes = [pool_list[i][1] for i in range(len(pool_list))]
    enemies_lifes = [pool_list[i][2] for i in range(len(pool_list))]

    fitnesses_row_per_sol = np.reshape(fitnesses, (int(len(pool_list) / repetitions), repetitions))
    player_lifes_row_per_sol = np.reshape(player_lifes, (int(len(pool_list) / repetitions), repetitions))
    enemies_lifes_row_per_sol = np.reshape(enemies_lifes, (int(len(pool_list) / repetitions), repetitions))

    individual_gain = player_lifes_row_per_sol - enemies_lifes_row_per_sol
    mean_individual_gain_model1 = np.mean(individual_gain, axis=1)

    mean_individual_gain_model1.sort()
    mean_individual_gain_model1 = mean_individual_gain_model1[-10:]

    ### --------------------------------- results 2 --------------------------------------- ###

    enemies = 8
    model = 'distanceroulette_weightedNpoint_adaptmutation'
    name = 'robin'

    # with open('best_sol_' + model + '_' + name, 'rb') as config_dictionary_file:
    #     config_df = pickle.load(config_dictionary_file)

    with open('best_results_1', 'rb') as config_df_file:
        config_df = pickle.load(config_df_file)

    repetitions = 5
    selected_df = config_df.loc[(config_df['enemies'] == enemies) & (config_df['model'] == model)]

    controllers = selected_df['best_solution'].values

    # create input including the number of neurons and the enemies so this isn't in the simulate function
    pool_input = [(pcont, [enemies], "no") for pcont in controllers for i in range(repetitions)]

    # run the simulations in parallel
    pool = Pool(cpu_count())
    pool_list = pool.starmap(play_game, pool_input)
    pool.close()
    pool.join()

    # get the fitnesses from the total results formatted as [(f, p, e, t), (...), ...]
    fitnesses = [pool_list[i][0] for i in range(len(pool_list))]
    player_lifes = [pool_list[i][1] for i in range(len(pool_list))]
    enemies_lifes = [pool_list[i][2] for i in range(len(pool_list))]

    fitnesses_row_per_sol = np.reshape(fitnesses, (int(len(pool_list) / repetitions), repetitions))
    player_lifes_row_per_sol = np.reshape(player_lifes, (int(len(pool_list) / repetitions), repetitions))
    enemies_lifes_row_per_sol = np.reshape(enemies_lifes, (int(len(pool_list) / repetitions), repetitions))

    individual_gain = player_lifes_row_per_sol - enemies_lifes_row_per_sol
    mean_individual_gain_model2 = np.mean(individual_gain, axis=1)

    mean_individual_gain_model2.sort()
    mean_individual_gain_model2 = mean_individual_gain_model2[-10:]

    levene_enemy_8 = levene(mean_individual_gain_model1, mean_individual_gain_model2)
    ttest_enemy_8 = ttest_ind(mean_individual_gain_model1, mean_individual_gain_model2)
    f_enemy_8 = f_oneway(mean_individual_gain_model1, mean_individual_gain_model2)


    ### --------------------------------- results 3 --------------------------------------- ###
    enemies = 7
    model = 'roulette_weightedNpoint_adapt'
    name = 'robin'

    with open('best_sol_' + model + '_' + name, 'rb') as config_dictionary_file:
        config_df = pickle.load(config_dictionary_file)

    repetitions = 5
    selected_df = config_df.loc[(config_df['enemies'] == enemies) & (config_df['model'] == model + '_' + name)]

    controllers = selected_df['best_solution'].values

    # create input including the number of neurons and the enemies so this isn't in the simulate function
    pool_input = [(pcont, [enemies], "no") for pcont in controllers for i in range(repetitions)]

    # run the simulations in parallel
    pool = Pool(cpu_count())
    pool_list = pool.starmap(play_game, pool_input)
    pool.close()
    pool.join()

    # get the fitnesses from the total results formatted as [(f, p, e, t), (...), ...]
    fitnesses = [pool_list[i][0] for i in range(len(pool_list))]
    player_lifes = [pool_list[i][1] for i in range(len(pool_list))]
    enemies_lifes = [pool_list[i][2] for i in range(len(pool_list))]

    fitnesses_row_per_sol = np.reshape(fitnesses, (int(len(pool_list) / repetitions), repetitions))
    player_lifes_row_per_sol = np.reshape(player_lifes, (int(len(pool_list) / repetitions), repetitions))
    enemies_lifes_row_per_sol = np.reshape(enemies_lifes, (int(len(pool_list) / repetitions), repetitions))

    individual_gain = player_lifes_row_per_sol - enemies_lifes_row_per_sol
    mean_individual_gain_model3 = np.mean(individual_gain, axis=1)

    mean_individual_gain_model3.sort()
    mean_individual_gain_model3 = mean_individual_gain_model3[-10:]

    ### --------------------------------- results 4 --------------------------------------- ###
    enemies = 7
    model = 'distanceroulette_weightedNpoint_adapt'
    name = 'robin'

    with open('best_sol_' + model + '_' + name, 'rb') as config_dictionary_file:
        config_df = pickle.load(config_dictionary_file)


    repetitions = 5
    selected_df = config_df.loc[(config_df['enemies'] == enemies) & (config_df['model'] == model + '_' + name)]

    controllers = selected_df['best_solution'].values

    # create input including the number of neurons and the enemies so this isn't in the simulate function
    pool_input = [(pcont, [enemies], "no") for pcont in controllers for i in range(repetitions)]

    # run the simulations in parallel
    pool = Pool(cpu_count())
    pool_list = pool.starmap(play_game, pool_input)
    pool.close()
    pool.join()

    # get the fitnesses from the total results formatted as [(f, p, e, t), (...), ...]
    fitnesses = [pool_list[i][0] for i in range(len(pool_list))]
    player_lifes = [pool_list[i][1] for i in range(len(pool_list))]
    enemies_lifes = [pool_list[i][2] for i in range(len(pool_list))]

    fitnesses_row_per_sol = np.reshape(fitnesses, (int(len(pool_list) / repetitions), repetitions))
    player_lifes_row_per_sol = np.reshape(player_lifes, (int(len(pool_list) / repetitions), repetitions))
    enemies_lifes_row_per_sol = np.reshape(enemies_lifes, (int(len(pool_list) / repetitions), repetitions))

    individual_gain = player_lifes_row_per_sol - enemies_lifes_row_per_sol
    mean_individual_gain_model4 = np.mean(individual_gain, axis=1)

    mean_individual_gain_model4.sort()
    mean_individual_gain_model4 = mean_individual_gain_model4[-10:]

    levene_enemy_7 = levene(mean_individual_gain_model3, mean_individual_gain_model4)
    ttest_enemy_7 = ttest_ind(mean_individual_gain_model3, mean_individual_gain_model4, equal_var=False)
    f_enemy_7 = f_oneway(mean_individual_gain_model3, mean_individual_gain_model4)


    ### --------------------------------- results 5 --------------------------------------- ###
    enemies = 5
    model = 'roulette_weightedNpoint_adapt'
    name = 'robin'

    with open('best_sol_' + model + '_' + name, 'rb') as config_dictionary_file:
        config_df = pickle.load(config_dictionary_file)

    repetitions = 5
    selected_df = config_df.loc[(config_df['enemies'] == enemies) & (config_df['model'] == model + '_' + name)]

    controllers = selected_df['best_solution'].values

    # create input including the number of neurons and the enemies so this isn't in the simulate function
    pool_input = [(pcont, [enemies], "no") for pcont in controllers for i in range(repetitions)]

    # run the simulations in parallel
    pool = Pool(cpu_count())
    pool_list = pool.starmap(play_game, pool_input)
    pool.close()
    pool.join()

    # get the fitnesses from the total results formatted as [(f, p, e, t), (...), ...]
    fitnesses = [pool_list[i][0] for i in range(len(pool_list))]
    player_lifes = [pool_list[i][1] for i in range(len(pool_list))]
    enemies_lifes = [pool_list[i][2] for i in range(len(pool_list))]

    fitnesses_row_per_sol = np.reshape(fitnesses, (int(len(pool_list) / repetitions), repetitions))
    player_lifes_row_per_sol = np.reshape(player_lifes, (int(len(pool_list) / repetitions), repetitions))
    enemies_lifes_row_per_sol = np.reshape(enemies_lifes, (int(len(pool_list) / repetitions), repetitions))

    individual_gain = player_lifes_row_per_sol - enemies_lifes_row_per_sol
    mean_individual_gain_model5 = np.mean(individual_gain, axis=1)

    mean_individual_gain_model5.sort()
    mean_individual_gain_model5 = mean_individual_gain_model5[-10:]


    ### --------------------------------- results 6 --------------------------------------- ###
    enemies = 5
    model = 'distanceroulette_weightedNpoint_adapt'
    name = 'robin'

    with open('best_sol_' + model + '_' + name, 'rb') as config_dictionary_file:
        config_df = pickle.load(config_dictionary_file)

    repetitions = 5
    selected_df = config_df.loc[(config_df['enemies'] == enemies) & (config_df['model'] == model + '_' + name)]

    controllers = selected_df['best_solution'].values

    # create input including the number of neurons and the enemies so this isn't in the simulate function
    pool_input = [(pcont, [enemies], "no") for pcont in controllers for i in range(repetitions)]

    # run the simulations in parallel
    pool = Pool(cpu_count())
    pool_list = pool.starmap(play_game, pool_input)
    pool.close()
    pool.join()

    # get the fitnesses from the total results formatted as [(f, p, e, t), (...), ...]
    fitnesses = [pool_list[i][0] for i in range(len(pool_list))]
    player_lifes = [pool_list[i][1] for i in range(len(pool_list))]
    enemies_lifes = [pool_list[i][2] for i in range(len(pool_list))]

    fitnesses_row_per_sol = np.reshape(fitnesses, (int(len(pool_list) / repetitions), repetitions))
    player_lifes_row_per_sol = np.reshape(player_lifes, (int(len(pool_list) / repetitions), repetitions))
    enemies_lifes_row_per_sol = np.reshape(enemies_lifes, (int(len(pool_list) / repetitions), repetitions))

    individual_gain = player_lifes_row_per_sol - enemies_lifes_row_per_sol
    mean_individual_gain_model6 = np.mean(individual_gain, axis=1)

    mean_individual_gain_model6.sort()
    mean_individual_gain_model6 = mean_individual_gain_model6[-10:]

    levene_enemy_5 = levene(mean_individual_gain_model5, mean_individual_gain_model6)
    ttest_enemy_5 = ttest_ind(mean_individual_gain_model5, mean_individual_gain_model6, equal_var=False)
    f_enemy_5 = f_oneway(mean_individual_gain_model5, mean_individual_gain_model6)


    plt.figure()
    plt.boxplot([mean_individual_gain_model1, mean_individual_gain_model2, mean_individual_gain_model3,
                 mean_individual_gain_model4, mean_individual_gain_model5, mean_individual_gain_model6])
    plt.xticks([1, 2, 3, 4, 5, 6], ['EA1 - E8', 'EA2 - E8', 'EA1 - E7', 'EA2 - E7', 'EA1 - E5', 'EA2 - E5'])
    plt.ylabel('Individual Gain', fontsize=14)
    plt.title('Best solutions of all runs')
    plt.grid()
    plt.savefig('best_solutions.png', dpi=300)
    plt.show()

    print("levene enemy 7: ", levene_enemy_7)
    print("levene enemy 8: ", levene_enemy_8)
    print("levene enemy 5: ", levene_enemy_5)


    print("t-test enemy 7: ", ttest_enemy_7)
    print("t-test enemy 8: ", ttest_enemy_8)
    print("t-test enemy 5: ", ttest_enemy_5)

    print("f enemy 7: ", f_enemy_7)
    print("f enemy 8: ", f_enemy_8)
    print("f enemy 5: ", f_enemy_5)
