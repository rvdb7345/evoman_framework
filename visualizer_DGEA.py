#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A script to visualize the data collected by the Monte Carlo simulations

Created on Thursday Sep 17 2020
This code was implemented by
Julien Fer, ...., ...., ....
"""

import os
import pandas as pd

class Visualizer(object):
    """
    Contains several methods to visualize the results found by the MC simulations
    """
    def __init__(self, name, csv_fitnesses_EA, csv_diversity_EA):
        self.results_EA = os.path.join("results", name)
        self.load_stats_EA(csv_fitnesses_EA, csv_diversity_EA)

    def load_stats_EA(self, csv_fitnesses_EA, csv_diversity_EA):
        """
        Load the statistics collected from evolutionary runs.
        """
        self.pd_fits_EA = pd.read_csv(os.path.join(self.results_EA, csv_fitnesses_EA))
        self.pd_div_EA = pd.read_csv(os.path.join(self.results_EA, csv_diversity_EA))
        print(self.pd_fits_EA)
        print(self.pd_div_EA)

if __name__ == "__main__":
    visualizer = Visualizer("dgea_test", "fitnesses_e7e8.csv", "diversity_e7e8.csv")