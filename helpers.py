#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script contains helper functions for retrieving arguments from user

Created on Thursday Sep 17 2020
This code was implemented by
Julien Fer, ...., ...., ....
"""

import argparse

# global variables for default values command-line arguments
SELECTIONS = ["random", "roulette", "distanceroulette"]
XOVERS = ["randomNpoint", "weightedNpoint", "randomlinear", "weightedlinear"]
MUTATIONS= ["normal", "scramble", "adapt"]
# MUTATIONS2 = ["scramble", "adapt"] #  under construction

def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def set_arguments():
    """
    Set command-line arguments
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=
            "This program contains several Gentetic Algorithms (GA) that can be "\
            "applied to the (specialist) Evomen Framework\n" 
    )
    parser.add_argument(
        "selection_and_surival",
        type=str, 
        default=SELECTIONS, 
        help="selection and survival method of GA, only if distance it is slightly different\n" \
            "default=[random, roulette, distance]"
    )
    parser.add_argument(
        "crossover",
        type=str, 
        default=XOVERS, 
        help="crossover method of GA\n(default=[randomNpoint, weightedNpoint, randomlinear, weightedlinear])"
    )
    parser.add_argument(
        "mutation",
        type=str, 
        default=MUTATIONS, 
        help="mutation method of GA\n(default=[normal, scramble, adapt])"
    )
    parser.add_argument(
        "name",
        type=str,
        help="name of user so that results can be stored in a personal file/folder"
    )
    parser.add_argument(
        "-layers", 
        type=int,
        default=1,
        help="amount of hidden layers in the neural network\n(default=1)"
    )
    parser.add_argument(
        "-neurons", 
        type=int,
        default=10,
        help="number of neurons in hidden layers of neual network\n(default=10)"
    )
    parser.add_argument(
        "-activation",
        type=str,
        action="append",
        default=["sigmoid"],
        help="type of activation function used (it can accept multiple types)\n" \
            "(default=[sigmoid])"
    )
    parser.add_argument(
        "-pop_size", 
        type=int,
        default=100,
        help="population size that starts GA\n(default=100)"
    )
    parser.add_argument(
        "-gens", 
        type=int,
        default=50,
        help="number of generations for GA\n(default=50)"
    )
    parser.add_argument(
        "-enemies",
        type=int,
        action="append",
        default=[],
        help="enemies against GA will be applied (can be multiple enemies)\n" \
            "(default=[8])"
    )
    parser.add_argument(
        "-N",
        type=int,
        default=10,
        help="amount of repetitions for genetic algorithm\n(default=10)"
    )
    parser.add_argument(
        "-tune", 
        type=str2bool,
        default=False, 
        help="parameter tuning for mutation probability\n" \
            "(default=False)"
    )
    parser.add_argument(
        "-show_plot", 
        type=str2bool,
        default=False, 
        help="shows plot of averages and standard devations for each run of GA\n" \
            "(default=False)"
    )
    parser.add_argument(
        "-show_endplot",
        type=str2bool,
        default=True,
        help="shows plot of averages and standard devations for end of monte carlo\n" \
            "(default=True)"
    )
    parser.add_argument(
        "-save_output", 
        type=str2bool,
        default=True, 
        help="shows plot of averages and standard devations for run of GA\n" \
            "(default=True)"
    )

    parser = parser.parse_args()
    return parser