#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A script to run monte carlo simulation with GA algorithms for the Evoman framwork

Created on Thursday Sep 17 2020
This code was implemented by
Julien Fer, ...., ...., ....
"""

import os
import shutil
import pandas as pd

class MonteCarlo(object):
    """
    OOP representation to run N simulations of a certain GA in the EvoMan
    framework. Note, we assume only one hidden layer for the NN.
    """
    def __init__(self, name, GA, N, save_output):
        self.name = name
        self.GA = GA
        self.N = N
        self.save_output = save_output
        self.results_folder = os.path.join("results", self.name)

        # remove old data if exists, and make sure a (new) results folder exists
        if os.path.exists(self.results_folder) and save_output:
            shutil.rmtree(self.results_folder)
        if save_output:
            os.makedirs(self.results_folder, exist_ok=True)

        # make csv filename for fitness and diversity
        enemies_str = ""
        for enemy in self.GA.enemies:
            enemies_str += "e" + str(enemy)
        self.csv_fitnesses = "fitnesses_" + enemies_str + ".csv"
        self.csv_best_fits = "best_fits_" + enemies_str + ".csv"
        self.csv_diversity = "diversity_" + enemies_str + ".csv"


    def save_stats(self, sim, fitnesses, best_fits, diversity_gens):
        """
        Save fitnesses and diversity colleted by GA during run to a file
        """
        
        filename = os.path.join(self.results_folder, self.csv_fitnesses)
        df_fitnesses = pd.DataFrame(fitnesses)
        if os.path.exists(filename):
            df_fitnesses.to_csv(filename, mode='a', header=False, index=False)
        else:
            df_fitnesses.to_csv(filename, mode='w', index=False)

        filename = os.path.join(self.results_folder, self.csv_best_fits)
        df_best_fits = pd.DataFrame(best_fits)
        if os.path.exists(filename):
            df_best_fits.to_csv(filename, mode='a', header=False, index=False)
        else:
            df_best_fits.to_csv(filename, mode='w', index=False)

        filename = os.path.join(self.results_folder, self.csv_diversity)
        df_diversity = pd.DataFrame(diversity_gens)
        if os.path.exists(filename):
            df_diversity.to_csv(filename, mode='a', header=False, index=False)
        else:
            df_diversity.to_csv(filename, mode="w", index=False)

    def run(self):
        """
        Run monte carlo simulation of EA
        """

        # start simulation
        for n in range(self.N):
            # enemy, best_fit, best_sol, generations_sum_df = self.GA.run()
            fitnesses, best_fits, diversities, best_fit, best_sol, total_exploit, total_explore = self.GA.run(n)

            # save statistics
            if self.save_output:
                self.save_stats(n, fitnesses, best_fits, diversities)

            # reset EA
            self.GA.reset_algorithm()