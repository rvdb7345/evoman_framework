#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A script to run monte carlo simulation with GA algorithms for the Evoman framwork

Created on Thursday Sep 17 2020
This code was implemented by
Julien Fer, ...., ...., ....
"""

import os
import pandas as pd

class MonteCarlo(object):
    """
    OOP representation to run N simulations of a certain GA in the EvoMan
    framework. Note, we assume only one hidden layer for the NN.
    """
    def __init__(self, name, GA, N, show_plot, save_output):
        self.name = name
        self.GA = GA
        self.N = N
        self.show_plot = show_plot
        self.save_output = save_output
        self.results_folder = os.path.join("results", self.name)

        os.makedirs(self.results_folder, exist_ok=True)

    def save_stats(self, sim, fitnesses, diversity_gens):
        """
        Save fitnesses and diversity colleted by GA during run to a file
        """
        enemies_str = ""
        for enemy in self.GA.enemies:
            enemies_str += "e" + str(enemy)
        csv_fitnesses = "fitnesses_" + enemies_str + ".csv"
        filename = os.path.join(self.results_folder, csv_fitnesses)
        df_fitnesses = pd.DataFrame(fitnesses)
        if os.path.exists(filename):
            df_fitnesses.to_csv(filename, mode='a', header=False)
        else:
            df_fitnesses.to_csv(filename, mode='w')

        csv_diversity = "diversity_" + enemies_str + ".csv"
        filename = os.path.join(self.results_folder, csv_diversity)
        df_diversity = pd.DataFrame(diversity_gens)
        if os.path.exists(filename):
            df_diversity.to_csv(filename, mode='a', header=False)
        else:
            df_diversity.to_csv(filename, mode="w")

    def run(self):
        """
        Run monte carlo simulation of EA
        """

        # start simulation
        for n in range(self.N):
            # enemy, best_fit, best_sol, generations_sum_df = self.GA.run()
            fitnesses, diversities, best_fit, best_sol, total_exploit, total_explore = self.GA.run(n)

            # save statistics
            if self.save_output:
                self.save_stats(n, fitnesses, diversities)

            # # show simple error plot
            # if self.show_plot:
            #     self.GA.simple_errorbar()

            # reset EA
            self.GA.reset_algorithm()