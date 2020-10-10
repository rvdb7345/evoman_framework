#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A script to analyse the multiple "optimal" solutions found by the Monte Carlo
simulations in the EvoMan framework.

Created on Saturday Oct 10 2020
This code was implemented by
Julien Fer, Robin van den Berg, ...., ....
"""

import os
import csv
import numpy as np

def load_best_sols(experiment_name, filename_sols, filename_fits):
    """
    Loads best solutions with their corresponding best fitness of 
    Monte Carlo simulations into a ??? . Also determines the best solution and 
    the one that differs the most from the best solution.
    """
    relpath_sols = os.path.join("results", experiment_name, filename_sols)
    relpath_fits = os.path.join("results", experiment_name, filename_fits)
    
    # load solutions and their fitnesses
    best_sols = np.load(relpath_sols)
    total_sols = len(best_sols)
    best_fits = []
    with open(relpath_fits, 'r') as csv_file:
        reader = csv.reader(csv_file)
        best_fits = [float(row[0]) for row in reader]
        best_fits = best_fits[1:]

    # find best solution
    best_fit_idx = np.argmax(best_fits)
    best_fit = best_fits[best_fit_idx]
    best_sol = best_sols["arr_" + str(best_fit_idx)]

    # find solution with max distance and with minimum difference in best fitness
    max_poss_distance = len(best_sol) * 2
    distances, max_distance, idx_max_distance = [], 0, 0
    difference_fits, min_diff_fit, idx_mindiff_fit = [], np.inf, 0
    for sol in range(total_sols):

        # determine distance and difference in fitness
        other_sol = best_sols["arr_" + str(sol)]
        distance = np.linalg.norm(best_sol - sol) / max_poss_distance
        other_fit = best_fits[sol]
        diff_fit = best_fit - other_fit

        # found "more different" solution
        if distance > max_distance:
            max_distance = distance
            idx_max_distance = sol

        # found different but relatively better solution w.r.t other nonbest solutios
        if diff_fit < min_diff_fit and sol != best_fit_idx:
            min_diff_fit = diff_fit
            idx_mindiff_fit = sol

        distances.append(distance), difference_fits.append(diff_fit)
    
    # most "different" solution
    sol_maxdistance = best_sols["arr_" + str(idx_max_distance)]
    
    # only keep the second best solution if it is different enough
    sol_mindiff_fit = None
    if distances[idx_mindiff_fit] > 0.5:
        sol_mindiff_fit = best_sols["arr_" + str(idx_mindiff_fit)]

    # returns best solution, most different solution and second best solution
    return best_sol, sol_maxdistance, sol_mindiff_fit

if __name__ == "__main__":
    filenames = ["newblood_random_training", "best_sols_e7e8.npz", "best_fits_best_solse7e8.csv"]
    best_sol, sol_maxdistance, sol_mindiff_fit = load_best_sols(*filenames)
    print("BEST SOLUTION")
    print(best_sol)
    print("==================================================================")
    print("MOST DIFFERENT SOLUTION")
    print(sol_maxdistance)
    print("==================================================================")
    print("SECOND BEST SOLUTION")
    print(sol_mindiff_fit)
    print("==================================================================")