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
from tqdm import tqdm as progressbar
from DGEA_julien import DGEA, NewBlood, NewBloodRandom, NewBloodDirected, NewBloodRandomElitism

class BasicGA(object):
    """
    Basic genetic algorithm to optimize certain parameters for the DGEA
    """
    def __init__(self, name, parameters, enemies, total_generations, save_output):
        """
        Initialize attributes
        """
        self.algorithm_name = name.split("_")[0]
        self.name = name + "_tuning"
        self.results_folder = os.path.join("results", self.name)
        if save_output:
            os.makedirs(self.results_folder, exist_ok=True)
        self.parameters = parameters
        self.total_generations = total_generations
        self.enemies = enemies
        self.save_output = save_output
        self.mutation_prob = 0.2

        # search intervals for parameters
        self.ranges_dmin = [[0, 0.05], [0.05, 0.1]]
        self.ranges_dmax = [0.3, 0.5]
        if self.algorithm_name == "dgea":
            self.ranges_mutation = [[0, 0.1], [0.1, 0.2]]
        else:
            self.ranges_mutation = [[0.5, 0.7], [0.7, 1]]
        self.individual_per_interval = 3

        # attributes to keep track of best fits and its corresponding parameters
        self.data = []
        self.best_fit, self.best_combo = None, None
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
        len_ranges_mutation = len(self.ranges_mutation)
        total_combos = self.individual_per_interval * (len_ranges_div + len_ranges_mutation)

        # semi-randomly sample initial (dmin, dmax, mutation factor) population
        population = np.zeros((total_combos, 3))
        for i in range(total_combos):

            # sample (dmin, dmax)
            idx_dmin = i % len_ranges_div
            dmin_lb, dmin_ub = self.ranges_dmin[idx_dmin]
            dmin = np.random.uniform(dmin_lb, dmin_ub)
            dmax = np.random.uniform(*self.ranges_dmax)
            population[i, 0], population[i, 1] = dmin, dmax

            # sample mutation factor
            idx_mutation = i % len_ranges_mutation
            fact_lb, fact_ub = self.ranges_mutation[idx_mutation]
            mutation_fact = np.random.uniform(fact_lb, fact_ub)
            population[i, 2] = mutation_fact

        return population
    
    def play(self, population, gen):
        """
        Play loop to run the EA for each individual (paramaeter combination). 
        Also, keeps track of the best fit found during the evolutionary run 
        per individual
        """

        # start run
        scores = np.zeros(population.shape[0])
        for i, (dmin, dmax, mutation) in progressbar(enumerate(population), desc="play loop parameters"):
            GA = None
            self.parameters["dmin"] = dmin
            self.parameters["dmax"] = dmax

            # choose right evolutionary algorithm to run
            if self.algorithm_name == "dgea":
                self.parameters["mutation_factor"] = mutation
                GA = DGEA(self.name, self.parameters, self.enemies)
            elif self.algorithm_name == "newblood":
                self.parameters["mutation_prob"] = mutation
                GA = NewBlood(self.name, self.parameters, self.enemies)
            elif self.algorithm_name == "newblood_directed":
                self.parameters["mutation_prob"] = mutation
                GA = NewBloodDirected(self.name, self.parameters, self.enemies)
            elif self.algorithm_name == "newblood_random":
                self.parameters["mutation_prob"] = mutation
                GA = NewBloodRandom(self.name, self.parameters, self.enemies)
            else:
                self.parameters["mutation_prob"] = mutation
                GA = NewBloodRandomElitism(self.name, self.parameters, self.enemies)

            # keep track of best fitness found during the evolutionary run of EA
            _, _, _, best_fit, _, _, _, _ = GA.run(gen)
            scores[i] = best_fit

        return scores
    
    def update_data(self, gen, population, scores):
        """
        Updates data (combosof parameters and its corresponding best fit) 
        for each generation
        """
        for (dmin, dmax, mutation), best_fit in zip(population, scores):
            data = {
                "generation": gen,
                "dmin": dmin,
                "dmax": dmax,
                "mutation": mutation,
                "best fitness": best_fit
            }
            self.data.append(data)
            
            # no best solution yet or significantly better solution found, 
            # if so (re)start collection of best solutions
            if self.best_fit is None or best_fit > 1.05 * self.best_fit:
                self.best_fit = best_fit
                self.best_combo = [(dmin, dmax, mutation)]
                self.best_fits = [best_fit]
                self.best_combos = [(dmin, dmax, mutation)]

            # better best solution found, so update best solution 
            # and determine if old solutions should be kept 
            elif best_fit > self.best_fit:
                self.best_fit = best_fit
                self.best_combo = (dmin, dmax, mutation)

                new_best_fits, new_best_combos = [self.best_fit], [self.best_combo]
                for old_fit, old_combo in zip(self.best_fits, self.best_combos):
                    difference = sum([abs(best - old) for best, old in zip((dmin, dmax, mutation), old_combo)])
                    if old_fit >= 0.95 * best_fit and difference > 0.05 * len(self.best_combo):
                        new_best_fits.append(old_fit), new_best_combos.append(old_combo)

                self.best_fits, self.best_combos = new_best_fits, new_best_combos

            # equally good solution found, but only keep track if they differ from
            # all previous solutions found
            elif best_fit >= 0.95 * self.best_fit:
                differences = []
                for (old_dmin, old_dmax, old_mutation) in self.best_combos:
                    difference = abs(dmin - old_dmin) + abs(dmax - old_dmax) + abs(mutation - old_mutation)
                    differences.append(difference)

                if min(differences) > 0.05 * len(self.best_combo):
                    self.best_fits.append(best_fit)
                    self.best_combos.append((dmin, dmax, mutation))
                

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
                child += np.random.normal(0, 0.25)

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
        Normalize scores between 0 and 1 (x - min) / (max - min), negative 
        values are set to a "minimum" positive value
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
        Save data to csv file
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