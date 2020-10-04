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

# helper function to set command-line argument
from helpers_DGEA import set_arguments, collect_parameters

# import genetic alorithms (GA) and monte carlo
from DGEA_julien import DGEA
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

    # if parser.tune:
    #     tuner = BasicGA(experiment_name, parameters, parser.enemies, 30)
    #     tuner.run()

    if parser.algorithm == "dgea":
        GA = DGEA(experiment_name, parameters, parser.enemies)
        
    MC = MonteCarlo(experiment_name, GA, parameters["N"], parser.save_output)
    MC.run()

    visualizer = Visualizer(
        experiment_name, parameters["total_generations"], GA.enemies, MC.csv_fitnesses, 
        MC.csv_best_fits, MC.csv_diversity, parser.show_plot, parser.save_output
    )