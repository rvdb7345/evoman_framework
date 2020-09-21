#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A script to run monte carlo simulation with GA algorithms for the Evoman framwork

Created on Thursday Sep 17 2020
This code was implemented by
Julien Fer, ...., ...., ....
"""

import os
import pickle

import  pandas as pd

class Monte_Carlo(object):
    """
    """
    def __init__(self, name, GA, N, show_plot, show_endplot, save_output):
        """
        """
        self.name = name
        self.GA = GA
        self.N = N
        self.show_plot = show_plot
        self.show_endplot = show_endplot
        self.save_output = save_output

        print("Do we try to save the output? : ", self.save_output)

    def save_generations(self, generation_sum_df):
        if os.path.exists(os.path.join(os.path.abspath(''), 'generational_summary')):
            with open(os.path.join(os.path.abspath(''), 'generational_summary'), 'rb') as config_df_file:
                config_df = pickle.load(config_df_file)
                generation_sum_df = pd.concat([generation_sum_df, config_df])

        with open('generational_summary', 'wb') as config_dictionary_file:
            pickle.dump(generation_sum_df, config_dictionary_file)

    def save_best_solution(self, enemies, best_fit, sol):
        best_solution_df = pd.DataFrame({'model': self.name, 'enemies': enemies,
                                         'fitness': best_fit, 'best_solution': sol}, index=[0])

        if os.path.exists(os.path.join(os.path.abspath(''), 'best_results')):
            with open(os.path.join(os.path.abspath(''), 'best_results'), 'rb') as config_df_file:
                config_df = pickle.load(config_df_file)
                best_solution_df = pd.concat([best_solution_df, config_df], ignore_index=True)

        with open('best_results', 'wb') as config_dictionary_file:
            pickle.dump(best_solution_df, config_dictionary_file)

    def run(self):
        """
        Run monte carlo simulation of EA
        """

        # start simulation
        for n in range(self.N):
            enemy, best_fit, best_sol, generations_sum_df = self.GA.run_evolutionary_algo()

            # save statistics
            if self.save_output:
                self.save_best_solution(enemy, best_fit, sol)
                self.save_generations(generation_sum_df)

            # show simple error plot
            if self.show_plot:
                self.GA.simple_errorbar()

            # reset EA
            self.GA.reset_algorithm()