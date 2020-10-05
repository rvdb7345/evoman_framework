#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A script to visualize the data collected by the Monte Carlo simulations

Created on Thursday Sep 17 2020
This code was implemented by
Julien Fer, Robin van den Berg, ...., ....
"""

import os
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm as progressbar
from DGEA_julien import DGEA

class BasicGA(object):
    """
    Basic genetic algorithm to optimize certain parameters for the DGEA
    """
    def __init__(self, name, parameters, enemies, total_generations, save_output):
        """
        Initialize attributes
        """
        self.name = name + "_tuning"
        self.results_folder = os.path.join("results", self.name)
        if os.path.exists(self.results_folder) and save_output:
            shutil.rmtree(self.results_folder)
        if save_output:
            os.makedirs(self.results_folder, exist_ok=True)
        self.parameters = parameters
        self.total_generations = total_generations
        self.enemies = enemies
        self.save_output = save_output
        self.mutation_prob = 0.2

        # search intervals for parameters
        self.ranges_dmin = [[0, 0.15], [0.15, 0.30]]
        self.ranges_dmax = [0.5, 1]
        self.ranges_mutation_factor = [[0, 0.15], [0.15, 0.30]]
        self.individual_per_interval = 3

        # attributes to keep track of best fits and its corresponding parameters
        self.data = []
        self.best_fits, self.best_combos = [], []

        # make csv filename for fitness and diversity
        enemies_str = ""
        for enemy in self.enemies:
            enemies_str += "e" + str(enemy)
        self.csv_fitnesses = "best_fits_" + enemies_str + ".csv"

    def make_initial_population(self):
        """
        Intialize random combinations of parameters as population. Note, that it
        tries to "equally" distributed the parameters combinations
        """
        # set up initial "search spaces" for better distributed sampling
        len_ranges_div = len(self.ranges_dmin)
        len_ranges_mutation = len(self.ranges_mutation_factor)
        total_combos = self.individual_per_interval * (len_ranges_div + len_ranges_mutation)

        # semi-randomly sample initial (dmin, dmax, mutation factor) population
        population = np.zeros((total_combos, 3))
        for i in range(total_combos):

            # sample (dmin, dmax)
            idx_dmin = i % len_ranges_div
            dmin_lb, dmin_ub = self.ranges_dmin[idx_dmin]
            dmin = np.random.uniform(dmin_lb, dmin_ub)
            dmax_ub = self.ranges_dmax[idx_dmin]
            dmax = np.random.uniform(dmin + 1e-5, dmax_ub)
            population[i, 0], population[i, 1] = dmin, dmax

            # sample mutation factor
            idx_mutation = i % len_ranges_mutation
            fact_lb, fact_ub = self.ranges_mutation_factor[idx_mutation]
            mutation_fact = np.random.uniform(fact_lb, fact_ub)
            population[i, 2] = mutation_fact

        return population
    
    def play(self, population, gen):
        scores = np.zeros(population.shape[0])
        for i, (dmin, dmax, mutation_factor) in progressbar(enumerate(population), desc="play loop parameters"):
            self.parameters["dmin"] = dmin
            self.parameters["dmax"] = dmax
            self.parameters["mutation_factor"] = mutation_factor
            GA = DGEA(self.name, self.parameters, self.enemies)
            _, _, _, best_fit, _, _, _, _ = GA.run(gen)
            scores[i] = best_fit

        return scores
    
    def update_data(self, gen, population, scores):
        """
        Updates data (combosof parameters and its corresponding best fit) 
        for each generation
        """
        for (dmin, dmax, mutation_factor), best_fit in zip(population, scores):
            data = {
                "generation": gen,
                "dmin": dmin,
                "dmax": dmax,
                "mutation factor": mutation_factor,
                "best fitness": best_fit
            }
            self.data.append(data)
            
            # keep track of best solution
            if len(self.best_fits) == 0:
                self.best_fits = [best_fit]
                self.best_combos = [(dmin, dmax, mutation_factor)]
            elif best_fit > self.best_fits[0]:
                self.best_fits = [best_fit]
                self.best_combos = [(dmin, dmax, mutation_factor)]
            elif best_fit == self.best_fits[0]:
                for (x, y, z) in self.best_combos:
                    if x - dmin != 0 or y - dmax != 0 or z - mutation_factor != 0:
                        self.best_fits.append(best_fit)
                        self.best_combos.append((dmin, dmax, mutation_factor))

    def tournament(self, population, scores):
        """
        Performs binary tournament
        """
        idx1 = np.random.randint(0, population.shape[0])
        idx2 = np.random.randint(0, population.shape[0])

        if scores[idx1] > scores[idx2]:
            return population[idx1]

        return population[idx2]

    def crossover(self, population, scores):
        """
        Perfroms a linear crossover with the weights sampled from a uniform
        distribution. Note that it always ensures that dmin < dmax
        """

        # start creating offspring
        all_offspring = np.zeros((0, population.shape[1]))
        for offspring in range(population.shape[0]):
            parent1 = self.tournament(population, scores)
            parent2 = self.tournament(population, scores)
            weigth = np.random.uniform()
            child = (1 - weigth) * parent1 + weigth * parent2

            # mutation
            if np.random.uniform() < self.mutation_prob:
                child += np.random.normal()

            # make sure variables represent probalities (interval (0, 1))
            child = np.clip(child, 0.0000001, 0.9999999)

            # make sure dmin is smaller than dmax
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

    def save_data(self):
        """
        Save data to csv  file
        """
        filename = os.path.join(self.results_folder, self.csv_fitnesses)
        df_data = pd.DataFrame(self.data)
        df_data.to_csv(filename, mode='w', index=False)

    def run(self):
        """
        Run parameter tuning with help of a simple genetic algoritm
        """

        # run first population
        gen = 0
        population = self.make_initial_population()
        pop_size = population.shape[0]
        scores = self.play(population, gen)
        self.update_data(gen, population, scores)

        # start evolutionary algorithm
        for gen in progressbar(range(1, self.total_generations + 1), desc="evolutionary loop Tuner"):

            # make offspring, evaluate and keep track of data
            offspring = self.crossover(population, scores)
            scores_offspring = self.play(offspring, gen)
            self.update_data(gen, offspring, scores_offspring)
            population = np.vstack((population, offspring))
            scores = np.append(scores, scores_offspring)

            # best score and best combo
            best_score_idx = np.argmax(scores)
            # best_combo = population[best_score]

            # roulette wheel survival selection (no replacement), with 1 individual elitism
            norm_scores = self.normalise_scores(scores)
            probs = norm_scores / norm_scores.sum()
            chosen = np.random.choice(population.shape[0], pop_size, p=probs, replace=False)
            chosen = np.append(chosen[1:], best_score_idx)
            population = population[chosen]
            scores = scores[chosen]

        if self.save_output:
            self.save_data()

        return self.best_fits, self.best_combos