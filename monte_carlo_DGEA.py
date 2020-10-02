#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A script to run monte carlo simulation with GA algorithms for the Evoman framwork

Created on Thursday Sep 17 2020
This code was implemented by
Julien Fer, ...., ...., ....
"""

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

    def run(self):
        """
        Run monte carlo simulation of EA
        """

        # start simulation
        for n in range(self.N):
            # enemy, best_fit, best_sol, generations_sum_df = self.GA.run()
            self.GA.run()

            # # save statistics
            # if self.save_output:
            #     self.save_best_solution(enemy, best_fit, best_sol)
            #     self.save_generations(generations_sum_df)

            # show simple error plot
            if self.show_plot:
                self.GA.simple_errorbar()

            # reset EA
            # self.GA.reset_algorithm()