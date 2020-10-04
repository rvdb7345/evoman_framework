#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script contains helper functions for retrieving arguments from user

Created on Thursday Sep 17 2020
This code was implemented by
Julien Fer, ...., ...., ....
"""

import sys
import argparse

ALGORITHMS = ["dgea"]

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
            "applied to the (generalist) Evoman Framework\n" 
    )
    parser.add_argument(
        "algorithm",
        type=str, 
        default=ALGORITHMS, 
        help="genetic algorithm to use for neurovultion process (default=['dgea'])"
    )
    parser.add_argument(
        "name",
        type=str,
        help="personal name so results are identified"
    )
    parser.add_argument(
        "-enemies",
        type=int,
        action="append",
        default=[],
        help="enemies against GA will be applied (can be multiple enemies)\n" \
            "(default=[])"
    )
    parser.add_argument(
        "-tune",
        type=str2bool,
        default=False,
        help="runs GA for optimizing dmin and dmax of DGEA\n(default=False)"
    )
    parser.add_argument(
        "-show_plot", 
        type=str2bool,
        default=False, 
        help="shows plot of averages and standard devations for each run of GA\n" \
            "(default=False)"
    )
    parser.add_argument(
        "-save_output", 
        type=str2bool,
        default=True, 
        help="shows plot of averages and standard devations for run of GA\n" \
            "(default=True)"
    )

    parser = parser.parse_args()

    if len(parser.enemies) < 2:
        sys.exit("The number of enemies should be larger then 2")

    return parser

def collect_parameters(filename):
    """
    Collects parameters from file into a dictionary
    """
    parameters = {}
    with open(filename, 'r') as f:
        line = f.readline().rstrip()
        while line:
            [param, value] = line.split('=')
            if '.' in value:
                parameters[param] = float(value)
            else:
                parameters[param] = int(value)

            line = f.readline().rstrip()
    return parameters