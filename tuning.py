#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A script to tune certain parameters of a GA algorithms for the Evoman framwork.
Mainly focussed on mutation probability as most of the other parameters are 
dependent on the development of the EA (adaptive). Perhaps the SKIP_PARENTS can 
be tuned for a small amount of integers. Can also tune pop size (or not for 
comparison reasons???)

Created on Thursday Sep 17 2020
This code was implemented by
Julien Fer, ...., ...., ....
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from tqdm import tqdm

def gridsearch_linearNpoint(algorithm, filename, *params):
    """
    Tune parameters for GA version (Npoint, linear and distance) weighte or random.
    Tuning is done with an exhaustive grid search
    """
    name, inputs, layers, neurons, outputs, activation, activation_distr, LB, UB, pop_size, gens, enemies, multiplemode, replacement = params

    # parameters to be tested
    skip_parents = list(range(1, 2, 1))
    len_parents = len(skip_parents)
    mutation_probs = [x * 0.1 for x in range(0, 11, 2)]
    len_mutations = len(mutation_probs)

    # set amount of simulations per combo
    simulations = 2

    # keep track of mean fitness and its standard deviation
    mean_best_fitnesses = np.zeros((len_parents, len_mutations))
    std_best_fitnesses = np.zeros((len_parents, len_mutations))
    # std_fitnesses = np.zeros((len_repl, len_parents, len_mutations, simulations))
    
    # exhaustive grid search of all parameters combos
    for i, skip_parent in enumerate(skip_parents):
        for j, mutation_prob in enumerate(mutation_probs):

            # each parameter combo is run for 5 simulations, keep track of best fit
            best_fitnesses = np.zeros((len_parents, len_mutations, simulations))
            for sim in tqdm(range(simulations)):
                GA = algorithm(
                    name=name,
                    nr_inputs=inputs,
                    nr_layers=layers,
                    nr_neurons=neurons,
                    nr_outputs=outputs,
                    activation_func=activation,
                    activation_distr=activation_distr,
                    lower_bound=LB,
                    upper_bound=UB,
                    pop_size=pop_size,
                    nr_gens=gens,
                    mutation_chance=mutation_prob,
                    nr_skip_parents=skip_parent,
                    enemies=enemies,
                    multiplemode=multiplemode,
                    replacement=replacement
                )

                # run algorithm
                _, best_fit, _, _ = GA.run_evolutionary_algo()

                # update statistics
                best_fitnesses[i, j, sim] = best_fit

                # reset EA
                GA.reset_algorithm()

            # determine averages best fit and its standard deviation
            mean_best_fitnesses[i, j] = best_fitnesses[i, j, :].mean()
            std_best_fitnesses[i, j] = best_fitnesses[i, j, :].std()
                

    # make sure folder exists
    filename_fit = os.path.join("results", filename + "_fitness")
    os.makedirs(filename_fit, exist_ok=True)
    filename_std = os.path.join("results", filename + "_std")
    os.makedirs(filename, exist_ok=True)

    # find best parameters settings (by creating 3d surface plots)
    x_vec, y_vec = np.array(skip_parents), np.array(mutation_probs)
    # z_vec = mean_best_fitnesses.flatten()
    make_3Dsurface_plot(x_vec, y_vec, mean_best_fitnesses, filename_fit, "Average best fitness")
    make_3Dsurface_plot(x_vec, y_vec, std_best_fitnesses, filename_std, "Standard deviation fitness")

    # x_vec, y_vec = np.array(skip_parents), np.array(mutation_probs)
    # for i, replacement in enumerate(replacements):
    #     name_plot = filename + "_" + str(replacement)
    #     z_vec = mean_best_fitnesses[i, :, :]
    #     make_3Dsurface_plot(x_vec, y_vec, z_vec, name_plot)

def make_3Dsurface_plot(x_vec, y_vec, z_vec, filename, title):
    """
    """

    # create figure and add axis
    fig = plt.figure(figsize=(8,6))
    ax = plt.subplot(111, projection="3d")

    # create meshgrid
    X, Y = np.meshgrid(y_vec, x_vec)

    # plot surface
    plot = ax.plot_surface(X=X, Y=Y, Z=z_vec, cmap='YlGnBu_r', vmin=0, vmax=200)

    # adjust plot view
    ax.view_init(elev=50, azim=225)
    ax.dist=11

    # Set tick marks
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
    ax.zaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))

    # Set axis labels
    ax.set_xlabel("Skip Parents", labelpad=20)
    ax.set_ylabel(r"$p_{m}$", labelpad=20)
    ax.set_zlabel("{}".format(title), labelpad=20)

    plt.savefig(filename + ".pdf", dpi=300)
    plt.show()