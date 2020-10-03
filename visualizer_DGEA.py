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
    def __init__(self, name, csv_fitnesses_EA, csv_diversity_EA):
        self.results_EA = os.path.join("results", name)
        self.pd_fits_EA = pd.read_csv(os.path.join(self.results_EA, csv_fitnesses_EA))
        self.pd_div_EA = pd.read_csv(os.path.join(self.results_EA, csv_diversity_EA))
        self.plot_fits_EA()

    def plot_fits_EA(self):
        """
        Plots mean fits EA across the generations with its confidence interval
        """

        # first determine mean fitness per generation
        mean_fitnesses = self.pd_fits_EA.groupby("generation")["fitness"].mean()
        stds_fitnesses = self.pd_fits_EA.groupby("generation")["fitness"].std()
        lower_ci = mean_fitnesses - stds_fitnesses
        upper_ci = mean_fitnesses + stds_fitnesses

        # plot mean fitness across the generations
        plt.figure()
        generations = list(range(0, len(mean_fitnesses)))
        plt.plot(generations, mean_fitnesses, color="b")
        plt.fill_between(generations, lower_ci, upper_ci, color="blue", alpha=0.1)
        plt.xlabel('Generation (#)', fontsize=12)
        plt.ylabel('Fitness', fontsize=12)
        plt.show()

        # determine mean diversity per generation
        mean_diversity = self.pd_div_EA.groupby("generation")["diversity"].mean()
        stds_diversity = self.pd_div_EA.groupby("generation")["diversity"].std()
        lower_ci = mean_diversity - stds_diversity
        upper_ci = mean_diversity + stds_diversity

        # plot mean diversity across the generations
        plt.figure()
        plt.plot(generations, mean_diversity, color="b")
        plt.fill_between(generations, lower_ci, upper_ci, color="blue", alpha=0.1)
        plt.xlabel('Generation (#)', fontsize=12)
        plt.ylabel('Diversity', fontsize=12)
        plt.show()


if __name__ == "__main__":
    visualizer = Visualizer("dgea_robin", "fitnesses_e7e8.csv", "diversity_e7e8.csv")