#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A script to visualize the data collected by the Monte Carlo simulations

Created on Thursday Sep 17 2020
This code was implemented by
Julien Fer, Robin van den Berg, ...., ....
"""

import os
import pandas as pd
from matplotlib import pyplot as plt

class Visualizer(object):
    """
    Contains several methods to visualize the results found by the MC simulations
    """
    def __init__(self, name, max_generations, enemies, csv_fitnesses_EA, csv_best_fits, csv_diversity_EA, show_plot, save_plot):
        self.max_generations = max_generations
        self.enemies = enemies
        self.show_plot, self.save_plot = show_plot, save_plot

        # retrieve data from results folder
        self.results_EA = os.path.join("results", name)
        self.pd_fits_EA = pd.read_csv(os.path.join(self.results_EA, csv_fitnesses_EA))
        self.pd_bestfits_EA = pd.read_csv(os.path.join(self.results_EA, csv_best_fits))
        self.pd_div_EA = pd.read_csv(os.path.join(self.results_EA, csv_diversity_EA))

        # plots average monte carlo of mean and max fitness
        self.plot_fits_EA()

    def plot_fits_EA(self):
        """
        Plots mean fits EA across the generations with its confidence interval
        """

        # first determine if there are missing averages due to early convergence
        # if so extend with last mean value found
        fitness_sims_gens = self.pd_fits_EA.groupby(["simulation", "generation"]).mean()
        for name, group in fitness_sims_gens.groupby("simulation"):
            missing_values = self.max_generations + 1 - group["fitness"].size
            for missing_gen in range(group["fitness"].size, self.max_generations + 1):
                fitness_sims_gens.loc[(name, missing_gen), :] = fitness_sims_gens.loc[(name, group["fitness"].size - 1)]
            
        # determine mean fitness across the generations with confidence intervals
        fitness_gens = fitness_sims_gens.groupby("generation")
        mean_fitnesses = fitness_gens["fitness"].mean()
        stds_fitnesses = fitness_gens["fitness"].std()
        lower_ci = mean_fitnesses - stds_fitnesses
        upper_ci = mean_fitnesses + stds_fitnesses

        # first determine if there are missing values (max fitness)
        best_fits_sim = self.pd_bestfits_EA.groupby("simulation")
        for sim, group in best_fits_sim:
            missing_values = self.max_generations + 1 - group["best fit"].size
            for missing_gen in range(group["best fit"].size, self.max_generations + 1):
                prev_bestfit = self.pd_bestfits_EA[(self.pd_bestfits_EA["simulation"] == sim) 
                                            & (self.pd_bestfits_EA["generation"] == missing_gen - 1)]
                prev_bestfit = prev_bestfit["best fit"]
                df = pd.DataFrame([[sim, missing_gen, prev_bestfit.iloc[0]]], 
                                    columns=["simulation", "generation", "best fit"])
                self.pd_bestfits_EA = self.pd_bestfits_EA.append(df, ignore_index=True)

        # determine mean max fitness per generation with confidenc intervals
        mean_best_fits = self.pd_bestfits_EA.groupby("generation")["best fit"].mean()
        stds_best_fits = self.pd_bestfits_EA.groupby("generation")["best fit"].std()
        lower_ci_bestfits = mean_max_fits - stds_max_fits
        upper_ci_bestfits = mean_max_fits + stds_max_fits

        # plot mean and mean max fitness across the generations
        plt.figure()
        generations = mean_fitnesses.index
        plt.plot(generations, mean_fitnesses, color="b")
        plt.fill_between(generations, lower_ci, upper_ci, color="blue", alpha=0.1)
        plt.plot(generations, mean_best_fits, color="y", linestyle="--")
        plt.fill_between(generations, lower_ci_bestfits, upper_ci_bestfits, color="yellow", alpha=0.1)
        plt.title("Mean and max fitness across the generations")
        plt.xlabel("Generation (#)", fontsize=12)
        plt.ylabel("Fitness", fontsize=12)
        
        enemies_str = ""
        for enemy in self.enemies:
            enemies_str += "e" + str(enemy)
        filename = "mean_fits_DGEA_" + enemies_str + ".png"
        rel_path = os.path.join(self.results_EA, filename)

        if self.save_plot:
            plt.savefig(rel_path, dpi=300)
        if self.show_plot:    
            plt.show()

        plt.close()

        # first determine if there are missing values
        diversity_sims = self.pd_div_EA.groupby("simulation")
        for sim, group in diversity_sims:
            missing_values = self.max_generations + 1 - group["diversity"].size
            for missing_gen in range(group["diversity"].size, self.max_generations + 1):
                prev_div = self.pd_div_EA[(self.pd_div_EA["simulation"] == sim) 
                                            & (self.pd_div_EA["generation"] == missing_gen - 1)]
                prev_div = prev_div["diversity"]
                df = pd.DataFrame([[sim, missing_gen, prev_div.iloc[0]]], 
                                    columns=["simulation", "generation", "diversity"])
                self.pd_div_EA = self.pd_div_EA.append(df, ignore_index=True)

        # determine mean diversity per generation
        mean_diversity = self.pd_div_EA.groupby("generation")["diversity"].mean()
        stds_diversity = self.pd_div_EA.groupby("generation")["diversity"].std()
        lower_ci = mean_diversity - stds_diversity
        upper_ci = mean_diversity + stds_diversity

        # plot mean diversity across the generations
        plt.figure()
        plt.title("Mean diversity across the generations")
        plt.plot(generations, mean_diversity, color="b")
        plt.fill_between(generations, lower_ci, upper_ci, color="blue", alpha=0.1)
        plt.xlabel("Generation (#)", fontsize=12)
        plt.ylabel("Diversity", fontsize=12)

        filename = "mean_diversity_" + enemies_str + ".png"
        rel_path = os.path.join(self.results_EA, filename)
        if self.save_plot:
            plt.savefig(rel_path, dpi=300)
        if self.show_plot:
            plt.show()
        plt.close()


if __name__ == "__main__":
#     # visualizer = Visualizer("dgea_robin", [7, 8], "fitnesses_e7e8.csv", "diversity_e7e8.csv", True, True)
    visualizer = Visualizer("dgea_test", 5, [7, 8], "fitnesses_e7e8.csv", "best_fits_e7e8.csv", "diversity_e7e8.csv", True, True)
#     # visualizer = Visualizer("dgea_test_bigger", [7, 8], "fitnesses_e7e8.csv", "diversity_e7e8.csv", True, True)
