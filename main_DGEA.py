#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A script to run different versions of GA algorithms for the Evoman framwork

Created on Thursday Sep 17 2020
This code was implemented by
Julien Fer, ...., ...., ....
"""


# built-in modules
import sys, os
import random

# helper function to set command-line argument
from helpers_DGEA import set_arguments, collect_parameters

# import genetic alorithms (GA) and monte carlo
from DGEA_julien import DGEA, NewBlood, NewBloodDirected
from monte_carlo_DGEA import MonteCarlo

# import visualizer for monte carlo results
from visualizer_DGEA import Visualizer

# import parameter tuner
from tuner_dgea import BasicGA

os.environ["SDL_VIDEODRIVER"] = "dummy"

if __name__ == "__main__":
    
    # get command-line arguments, set experiment name and filename parameters
    parser = set_arguments()
    experiment_name = parser.algorithm + "_" + parser.name
    filename = "parameters_" + parser.algorithm + ".txt"

    parameters = collect_parameters(filename)

    # tune a few parameters and choose best parameters combo at random (if more found)
    if parser.tune:
        gens = 10
        tuner = BasicGA(experiment_name, parameters, parser.enemies, gens, True)
        best_fits, best_parameters = tuner.run()
        best_combo = random.choice(best_parameters)
        parameters["dmin"], parameters["dmax"], parameters["mutation_factor"] = best_combo

    # choose algorithm
    if parser.algorithm == "dgea":
        GA = DGEA(experiment_name, parameters, parser.enemies)
    elif parser.algorithm == "newblood":
        GA = NewBlood(experiment_name, parameters, parser.enemies)
    elif parser.algorithm == "newblood_directed":
        GA = NewBloodDirected(experiment_name, parameters, parser.enemies)
    
    # run monte carlo simulation of GA
    MC = MonteCarlo(experiment_name, GA, parameters["N"], parser.save_output)
    MC.run()

    # plot results
    if parser.tune:
        visualizer = Visualizer(
            experiment_name, parameters["total_generations"], GA.enemies, MC.csv_fitnesses, 
            MC.csv_best_fits, MC.csv_diversity, parser.show_plot, parser.save_output, tuner.csv_fitnesses
        )
    else:
        visualizer = Visualizer(
            experiment_name, parameters["total_generations"], GA.enemies, MC.csv_fitnesses, 
            MC.csv_best_fits, MC.csv_diversity, parser.show_plot, parser.save_output
        )

    print("parameters after tuning are")
    print(parameters["dmin"], parameters["dmax"], parameters["mutation_factor"])
    print("best parameters are")
    print(best_combo)
    print("All best combos are")
    print(best_parameters)