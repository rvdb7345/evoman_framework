#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A script to visualize the data collected by the Monte Carlo simulations

Created on Thursday Sep 17 2020
This code was implemented by
Julien Fer, Robin van den Berg, ...., ....
"""

import os
import random
import pandas as pd
from DGEA_julien import DGEA

class BasicGA(object):
    """
    Basic genetic algorithm to optimize certain parameters for the DGEA
    """
    def __init__(self, name, parameters, enemies, total_generations):
        """
        Initialize attributes
        """
        self.name = name
        self.results_folder = os.path.join("results", name)
        self.paramaters = parameters
        self.total_generations = total_generations
        # self.n_hidden_neurons = parameters["neurons"]
        # self.lower_bound, self.upper_bound = parameters["lb"], parameters["ub"]
        # self.pop_size, self.total_generations = parameters["pop_size"], parameters["total_generations"]
        # self.mutation_prob = parameters["mutation_prob"]
        # self.mutation_frac = parameters["mutation_factor"] * abs(self.upper_bound - self.lower_bound)
        # self.crossover_prob = parameters["crossover_prob"]
        # self.fraction_replace = parameters["fraction_replace"]
        # self.max_no_improvements = parameters["max_no_improvements"]
        self.enemies = enemies

    def run(self):
        """
        Run parameter tuning with help of a simple genetic algoritm
        """

        # set up initial "search spaces" for better distributed sampling
        ranges_dmin = [[0, 0.1], [0.1, 0.2], [0.2, 0.3]]
        ranges_dmax = [0.5, 0.7, 1]
        len_ranges = len(ranges_dmin)
        total_combos = 5 * len_ranges

        # semi-randomly sample initial (dmin, dmax) combos (this is the population)
        combos = []
        for i in range(total_combos):
            i = i % len_ranges
            dmin_lb, dmin_ub = ranges_dmin[i]
            dmin = random.uniform(dmin_lb, dmin_ub)
            dmax_ub = ranges_dmax[i]
            dmax = random.uniform(dmin + 1e-5, dmax_ub)
            combos.append((dmin, dmax))

        print(combos)

        # start evolutionary algorithm
        for gen in range(self.total_generations):
            for (dmin, dmax) in combos:
                self.paramaters["dmin"] = dmin
                self.paramaters["dmax"] = dmax
                GA = DGEA(self.name, self.paramaters, self.enemies)
                fitnesses, _, best_fit, _, _, _ = GA.run(gen)
                df_fitnesses = pd.DataFrame(fitnesses)
                mean_fitness = df_fitnesses["fitness"].mean()
                print(mean_fitness)
                break
            break