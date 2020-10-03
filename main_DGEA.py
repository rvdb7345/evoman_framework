#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A script to run different versions of GA algorithms for the Evoman framwork

Created on Thursday Sep 17 2020
This code was implemented by
Julien Fer, ...., ...., ....
"""

# built-in modules
import sys

# helper function to set command-line argument
from helpers_DGEA import set_arguments, collect_parameters

# import genetic alorithms (GA) and monte carlo
from DGEA_julien import DGEA
from monte_carlo_DGEA import MonteCarlo

if __name__ == "__main__":
    
    # get command-line arguments, set experiment name and filename parameters
    parser = set_arguments()
    experiment_name = parser.algorithm + "_" + parser.name
    filename = "parameters_" + parser.algorithm + ".txt"

    parameters = collect_parameters(filename)

    if parser.algorithm == "dgea":
        GA = DGEA(parser.algorithm, parameters, parser.enemies)
        
    MC = MonteCarlo(experiment_name, GA, parameters["N"], parser.show_plot, parser.save_output)
    MC.run()