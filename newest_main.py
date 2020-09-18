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
from helpers import set_arguments

# import genetic algorithms
import ga_algorithms_npoint as NpointAlgos
import ga_algorithms_linear as LinearAlgos

# import monte carlo object (for optimization)
from monte_carlo import Monte_Carlo

# dictionary mapping name to algorithm object
ALGORITHMS = {
    "random_randomNpoint_normal": NpointAlgos.GA_random_Npoint,
    "random_randomlinear_normal": LinearAlgos.GA_randomlinear,
    "roulette_randomNpoint_normal": NpointAlgos.GA_roulette_randomNpoint,
    "roulette_randomlinear_normal":LinearAlgos.GA_roulette_randomlinear,
    "roulette_randomNpoint_scramble": NpointAlgos.GA_roulette_randomNpoint_scramblemutation,
    "roulette_randomlinear_scramble": LinearAlgos.GA_roulette_randomlinear_scramblemutation,
    "roulette_randomNpoint_adapt": NpointAlgos.GA_roulette_randomNpoint_adaptmutation,
    "roulette_randomlinear_adapt": LinearAlgos.GA_roulette_randomlinear_adaptmutation,
    "roulette_weightedNpoint_normal": NpointAlgos.GA_roulette_weightedNpoint,
    "roulette_weightedlinear_normal": LinearAlgos.GA_roulette_weightedlinear,
    "roulette_weightedNpoint_scramble": NpointAlgos.GA_roulette_weightedNpoint_scramblemutation,
    "roulette_weightedlinear_scramble": LinearAlgos.GA_roulette_weightedlinear_scrambledmutation,
    "roulette_weightedNpoint_adapt": NpointAlgos.GA_roulette_weightedNpoint_adaptmutation,
    "roulette_weightedlinear_adapt": LinearAlgos.GA_roulette_weightedlinear_adaptmutation,
    "roulette_weightedNpoint_adaptscramble": NpointAlgos.GA_roulette_weightedNpoint_adaptscramblemutation,
    "roulette_weightedlinear_adaptscramble": LinearAlgos.GA_roulette_weightedlinear_adaptscramblemutation,
    "distanceroulette_randomNpoint_normal": NpointAlgos.GA_distanceroulette_randomNpoint,
    "distanceroulette_randomlinear_normal":LinearAlgos.GA_distanceroulette_randomlinear,
    "distanceroulette_randomNpoint_scramble": NpointAlgos.GA_distanceroulette_randomNpoint_scramblemutation,
    "distanceroulette_randomlinear_scramble": LinearAlgos.GA_distanceroulette_randomlinear_scramblemutation,
    "distanceroulette_randomNpoint_adapt": NpointAlgos.GA_distanceroulette_randomNpoint_adaptmutation,
    "distanceroulette_randomlinear_adapt": LinearAlgos.GA_distanceroulette_randomlinear_adaptmutation,
    "distanceroulette_weightedNpoint_normal": NpointAlgos.GA_distanceroulette_weightedNpoint,
    "distanceroulette_weightedlinear_normal": LinearAlgos.GA_distanceroulette_weightedlinear,
    "distanceroulette_weightedNpoint_scramble": NpointAlgos.GA_distanceroulette_weightedNpoint_scramblemutation,
    "distanceroulette_weightedlinear_scramble": LinearAlgos.GA_distanceroulette_weightedlinear_scramblemutation,
    "distanceroulette_weightedNpoint_adapt": NpointAlgos.GA_distanceroulette_weightedNpoint_adaptmutation,
    "distanceroulette_weightedlinear_adapt": LinearAlgos.GA_distanceroulette_weightedlinear_adaptmutation,
    "distanceroulette_weightedNpoint_adaptscramble": NpointAlgos.GA_distanceroulette_weightedNpoint_adaptscramblemutation,
    "distanceroulette_weightedlinear_adaptscramble": LinearAlgos.GA_distanecroulette_weightedlinear_adaptscramblemutation
}

# constants for genetic algorithm
INPUTS = 20
OUTPUTS = 5
LB = -1
UB = 1
MUTATION_PROB = 0.2
SKIP_PARENTS = 1
REPLACEMENT = False

if __name__ == "__main__":

    # get command-line arguments and make name of algorithm and name experiment
    parser = set_arguments()
    algorithm_name = parser.selection_and_surival + "_" + parser.crossover + "_" + parser.mutation
    experiment_name = algorithm_name + "_" + parser.name

    # ensures name algoritm is valid
    if algorithm_name not in ALGORITHMS:
        sys.exit("algorithm not found --> run python newest_main.py -h to see more info")

    # determine remaining parameters for genetic algorithm and ensure they're valid
    nr_activefuncs = len(parser.activation)
    activation_distr = [1 / nr_activefuncs] * nr_activefuncs
    if sum(activation_distr) != 1:
        sys.exit("distribtion of the activation functions does not sum up to one")

    multiplemode = "no"
    if len(parser.enemies) > 1:
        multiplemode = "yes"

    # retrieve algorithm and initialize
    algorithm = ALGORITHMS[algorithm_name]
    GA = algorithm(
        name=algorithm_name,
        nr_inputs=INPUTS, 
        nr_layers=parser.layers, 
        nr_neurons=parser.neurons, 
        nr_outputs=OUTPUTS,
        activation_func=parser.activation, 
        activation_distr=activation_distr,
        lower_bound=LB, 
        upper_bound=UB, 
        pop_size=parser.pop_size, 
        nr_gens=parser.gens, 
        mutation_chance=MUTATION_PROB, 
        nr_skip_parents=SKIP_PARENTS,
        enemies=parser.enemies, 
        multiplemode = multiplemode,
        replacement = REPLACEMENT
    )
    # if not parser.show_plot:
    #     sys.exit("Show plot is False")
    # else:
    #     sys.exit("Show plot is True")
    MC = Monte_Carlo(
        experiment_name, 
        GA, 
        parser.N, 
        parser.show_plot, 
        parser.show_endplot, 
        parser.save_output
    )
    MC.run()