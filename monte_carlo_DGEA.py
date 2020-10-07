#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A script to run monte carlo simulation with GA algorithms for the Evoman framwork

Created on Thursday Sep 17 2020
This code was implemented by
Julien Fer, ...., ...., ....
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm as progressbar

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
        self.best_fit = None
        self.best_fits, self.best_sols = [], []
        self.results_folder = os.path.join("results", self.name)
        if save_output:
            os.makedirs(self.results_folder, exist_ok=True)

        # make csv filename for fitness and diversity
        enemies_str = ""
        for enemy in self.GA.enemies:
            enemies_str += "e" + str(enemy)
        self.csv_fitnesses = "fitnesses_" + enemies_str + ".csv"
        self.csv_best_fits = "best_fits_" + enemies_str + ".csv"
        self.csv_diversity = "diversity_" + enemies_str + ".csv"
        self.best_sols_name = "best_sols_" + enemies_str + ".npz"
        self.best_fits_sols_name = "best_fits_best_sols" + enemies_str + ".csv"

    def update_best_solutions(self, best_fits, best_solutions):
        """
        Keep track of the best solutions (configuration NN) over all the 
        Monte Carlo simulation
        """

        # for each possible new solution we check if we add it to our collection of best solutions
        for best_fit, best_sol in zip(best_fits, best_solutions):

            # new best solutions is significantly better, so "restart" collections of best solutions
            if self.best_fit is None or best_fit > 1.05 * self.best_fit:
                self.best_fit = best_fit
                self.best_fits = [best_fit]
                self.best_sols = [best_sol]

            # new best solution is bit better, so update best sol and determine
            # which old solutions we should keep
            elif best_fit > self.best_fit:
                self.best_fit = best_fit
                new_best_fits, new_best_sols = [best_fit], [best_sol]
                for old_fit, old_sol in zip(self.best_fits, self.best_sols):
                    distance = np.linalg.norm(best_sol - old_sol)

                    # only keep old solution of which the fitnesses are in range and
                    # which are "different" from the current best solution
                    if old_fit >= 0.95 * best_fit and distance / self.GA.L > 0.1:
                        new_best_fits.append(old_fit), new_best_sols.append(old_sol)
                
                self.best_fits, self.best_sols = new_best_fits, new_best_sols

            # only keep worse solutions if their fitness are within range and if 
            # different from current best solution
            elif best_fit >= 0.95 * self.best_fit:
                distances = [np.linalg.norm(best_sol - curr_best) for curr_best in self.best_sols]
                if min(distances) / self.GA.L > 0.1:
                    self.best_sols.append(solution)
                    self.best_fits.append(best_fit)

    def save_stats(self, sim, fitnesses, best_fits, diversity_gens):
        """
        Save fitnesses and diversity colleted by GA during run to a file
        """
        
        # save fitnesses across the generations
        filename = os.path.join(self.results_folder, self.csv_fitnesses)
        df_fitnesses = pd.DataFrame(fitnesses)
        if os.path.exists(filename) and sim != 0:
            df_fitnesses.to_csv(filename, mode='a', header=False, index=False)
        else:
            df_fitnesses.to_csv(filename, mode='w', index=False)

        # save best fits across the generations
        filename = os.path.join(self.results_folder, self.csv_best_fits)
        df_best_fits = pd.DataFrame(best_fits)
        if os.path.exists(filename) and sim != 0:
            df_best_fits.to_csv(filename, mode='a', header=False, index=False)
        else:
            df_best_fits.to_csv(filename, mode='w', index=False)
        
        # save diversity across the generations
        filename = os.path.join(self.results_folder, self.csv_diversity)
        df_diversity = pd.DataFrame(diversity_gens)
        if os.path.exists(filename) and sim != 0:
            df_diversity.to_csv(filename, mode='a', header=False, index=False)
        else:
            df_diversity.to_csv(filename, mode="w", index=False)

    def run(self):
        """
        Run monte carlo simulation of EA
        """
        try:
            # start simulation
            for n in progressbar(range(self.N), desc="monte carlo loop"):
                [
                    fitnesses, best_fit_gens, diversity_gens, best_fit, 
                    best_sol, best_fits, best_sols, total_exploit, total_explore
                ] = self.GA.run(n)
                

                # save statistics
                if self.save_output:
                    self.update_best_solutions(best_fits, best_sols)
                    self.save_stats(n, fitnesses, best_fit_gens, diversity_gens)

                # reset EA
                self.GA.reset_algorithm()

            # save best neural networks witth their best fitnesses
            if self.save_output:
                filename = os.path.join(self.results_folder, self.best_sols_name)
                np.savez(filename, *self.best_sols)
                filename = os.path.join(self.results_folder, self.best_fits_sols_name)
                df = pd.DataFrame(self.best_fits)
                df.to_csv(filename, mode='w', index=False)
            
        except KeyboardInterrupt:
            if self.save_output:
                filename = os.path.join(self.results_folder, self.best_sols_name)
                np.savez(filename, *self.best_sols)
                filename = os.path.join(self.results_folder, self.best_fits_sols_name)
                df = pd.DataFrame(self.best_fits)
                df.to_csv(filename, mode='w', index=False)