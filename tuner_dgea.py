#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A script to visualize the data collected by the Monte Carlo simulations

Created on Thursday Sep 17 2020
This code was implemented by
Julien Fer, Robin van den Berg, ...., ....
"""

import os
import numpy as np
import pandas as pd
from DGEA_julien import DGEA

class BasicGA(object):
    """
    Basic genetic algorithm to optimize certain parameters for the DGEA
    """
    def __init__(self, name, parameters, enemies, total_generations):
        """
        Initialize attributes
        """
        self.name = name
        self.results_folder = os.path.join("results", name)
        self.paramaters = parameters
        self.total_generations = total_generations
        # self.n_hidden_neurons = parameters["neurons"]
        # self.lower_bound, self.upper_bound = parameters["lb"], parameters["ub"]
        # self.pop_size, self.total_generations = parameters["pop_size"], parameters["total_generations"]
        # self.mutation_prob = parameters["mutation_prob"]
        # self.mutation_frac = parameters["mutation_factor"] * abs(self.upper_bound - self.lower_bound)
        # self.crossover_prob = parameters["crossover_prob"]
        # self.fraction_replace = parameters["fraction_replace"]
        # self.max_no_improvements = parameters["max_no_improvements"]
        self.enemies = enemies
    
    def play(self, population, gen):
        scores = np.zeros(population.shape[0])
        for i, (dmin, dmax) in enumerate(population):
            self.paramaters["dmin"] = dmin
            self.paramaters["dmax"] = dmax
            GA = DGEA(self.name, self.paramaters, self.enemies)
            fitnesses, _, _, best_fit, _, _, _ = GA.run(gen)
            df_fitnesses = pd.DataFrame(fitnesses)
            mean_fitness = df_fitnesses["fitness"].mean()
            scores[0] = mean_fitness

        return scores

    def tournament(self, population, scores):
        """
        Performs binary tournament
        """
        idx1 = np.random.randint(0, population.shape[0])
        idx2 = np.random.randint(0, population.shape[0])

        if scores[idx1] > score[idx2]:
            return population[idx1]

        return population[idx2]

    def crossover(self, population, scores, total_combos):
        """
        Perfroms a linear crossover with the weights sampled from a uniform
        distribution. Note that it always ensures that dmin < dmax
        """

        # start creating offspring
        all_offspring = np.zeros((0, 2))
        for offspring in range(total_combos):
            parent1 = self.tournament(population, scores)
            parent2 = self.tournament(population, scores)
            weigth = np.random.uniform()
            child = (1 - weight) * parent1 + weight * parent2
            
            if child[0] > child[1]:
                temp = child[0]
                child[0] = child[1]
                child[1] = temp
            
            all_offspring = np.vstack((all_offspring, child))

        return all_offspring

    def normalise_scores(self, scores):
        """
        Normalize scores to represent probabilites
        """
        norm_scores = np.zeros(len(scores))
        for i, score in enumerate(scores):
            if score - min(scores) > 0:
                norm = (score - min(scores)) / (max(scores) - min(scores))
            else:
                norm = 0

            if norm == 0:
                norm = 1e-10
            
            norm_scores[i] = norm
        
        return norm_scores

    def run(self):
        """
        Run parameter tuning with help of a simple genetic algoritm
        """

        # set up initial "search spaces" for better distributed sampling
        ranges_dmin = [[0, 0.1], [0.1, 0.2], [0.2, 0.3]]
        ranges_dmax = [0.5, 0.7, 1]
        len_ranges = len(ranges_dmin)
        total_combos = 5 * len_ranges

        # semi-randomly sample initial (dmin, dmax) combos (this is the population)
        population = np.zeros((total_combos, 2))
        for i in range(total_combos):
            i = i % len_ranges
            dmin_lb, dmin_ub = ranges_dmin[i]
            dmin = np.random.uniform(dmin_lb, dmin_ub)
            dmax_ub = ranges_dmax[i]
            dmax = np.random.uniform(dmin + 1e-5, dmax_ub)
            population[i, 0], population[i, 1] = dmin, dmax
            
        print(population)

        # run first population
        gen = 0
        scores = self.play(population, gen)

        # start evolutionary algorithm
        for gen in range(1, self.total_generations + 1):
            offspring = self.crossover(population, scores, total_combos)
            scores_offspring = self.play(offspring, gen)
            population = np.vstack((population, offspring))
            scores = np.append(scores, scores_offspring)

            # best score and best combo
            best_score = np.argmax(scores)
            best_combo = population[best_score]

            # selection
            norm_scores = self.normalise_scores(scores)
            probs = norm_scores / norm_scores.sum()
            chosen = np.random.choice(population.shape[0], total_combos, p=probs, replace=False)
            chosen = np.append(chosen[1, :], best_combo)
            