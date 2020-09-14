################################################################################
# Representation of a parallel simulation with as task to optimize an AI       #                                                        #
# 
# Names:
# University:
#                                                                              
################################################################################

# imports framework
import sys, os
sys.path.insert(0, "evoman")

import numpy as np

from simulation import SimulationRank, SimulationRoulette, SimulationWeightedRank
from simulation import SimulationAdaptiveMutationNpointCrossover

class DistanceRank(SimulationRank):
    def __init__(
            self, experiment_name, nr_inputs, nr_layers,
            nr_neurons, nr_outputs, activation_func, activation_distr,
            lower_bound, upper_bound, pop_size, nr_gens,
            mutation_chance, nr_skip_parents, enemies, multiplemode, min_dist_perc
        ):
        super().__init__(
            experiment_name, nr_inputs, nr_layers, nr_neurons,
            nr_outputs, activation_func, activation_distr,
            lower_bound, upper_bound, pop_size, nr_gens,
            mutation_chance, nr_skip_parents, enemies,  multiplemode
        )
        self.min_dist_perc = min_dist_perc

    def get_distance_probs(self, parent1, sorted_controls):
        """
        Determines the  probability based on the distance of the first
        parent selected relatively to that of all the others.
        """
        distances, params1 = [], parent1.get_params()
        for other_parent in sorted_controls:
            params2 = other_parent.get_params()
            distance = 0
            
            for layer in range(self.tot_layers):
                weight_str = "W" + str(layer)
                W1, W2 = params1[weight_str], params2[weight_str]
                diff = (W1 - W2).flatten()
                distance += np.sqrt(np.dot(diff, diff))

            distances.append(distance)

        # normalize, so we get probabilities
        # sum_distances = sum(distances)
        # distances_norm = [distance / sum_distances for distance in distances]
        min_distance, max_distance = min(distances), max(distances)
        range_distance = max_distance - min_distance
        distances_norm = [(distance - min_distance) / range_distance for distance in distances]

        return distances_norm

    def select_other_parent(self, fit_norm, sorted_controls, distances_norm, parent1):
        """
        Selects other parent and only accepts if the relative difference is 
        above a certain threshold.
        """

        tries, max_tries = 0, self.pop_size // self.nr_skip_parents
        other_parent = np.random.choice(sorted_controls, p=fit_norm)
        id_other = sorted_controls.index(other_parent)
        rel_distance = distances_norm[id_other]

        while rel_distance < self.min_dist_perc and tries < max_tries:
            other_parent = np.random.choice(sorted_controls, p=fit_norm)
            id_other = sorted_controls.index(other_parent)
            rel_distance = distances_norm[id_other]
            tries += 1

        return id_other, other_parent

    def make_new_generation(self, fit_norm_sorted, sorted_controls):
        """
        Crossover gense for a given population
        """

        # start creating children based on pairs of parents
        children = []
        for i in range(0, self.pop_size, self.nr_skip_parents):

            # rank selection of two parents
            _, parent1 = self.parent_selection(fit_norm_sorted, sorted_controls)
            distances_norm = self.get_distance_probs(parent1, sorted_controls)
            _, parent2 = self.select_other_parent(fit_norm_sorted, sorted_controls, distances_norm, parent1)

            # create child and add to children list
            child = self.crossover_and_mutation(parent1, parent2)
            children.append(child)

        # select parents based on normilzed probabilites of the fitnesses who
        # do not survive
        sorted_controls = self.determine_survival(fit_norm_sorted, sorted_controls, children)

        return sorted_controls

class DistanceRoulette(SimulationRoulette):
    def __init__(
            self, experiment_name, nr_inputs, nr_layers,
            nr_neurons, nr_outputs, activation_func, activation_distr,
            lower_bound, upper_bound, pop_size, nr_gens,
            mutation_chance, nr_skip_parents, enemies, multiplemode, min_dist_perc
        ):
        super().__init__(
            experiment_name, nr_inputs, nr_layers, nr_neurons,
            nr_outputs, activation_func, activation_distr,
            lower_bound, upper_bound, pop_size, nr_gens,
            mutation_chance, nr_skip_parents, enemies,  multiplemode
        )
        self.min_dist_perc = min_dist_perc

    def get_distance_probs(self, parent1, sorted_controls):
        """
        Determines the  probability based on the distance of the first
        parent selected relatively to that of all the others.
        """
        distances, params1 = [], parent1.get_params()
        for other_parent in sorted_controls:
            params2 = other_parent.get_params()
            distance = 0
            
            for layer in range(self.tot_layers):
                weight_str = "W" + str(layer)
                W1, W2 = params1[weight_str], params2[weight_str]
                diff = (W1 - W2).flatten()
                distance += np.sqrt(np.dot(diff, diff))

            distances.append(distance)

        # normalize, so we get probabilities
        # sum_distances = sum(distances)
        # distances_norm = [distance / sum_distances for distance in distances]
        min_distance, max_distance = min(distances), max(distances)
        range_distance = max_distance - min_distance
        distances_norm = [(distance - min_distance) / range_distance for distance in distances]

        return distances_norm

    def select_other_parent(self, fit_norm, sorted_controls, distances_norm, parent1):
        """
        Selects other parent and only accepts if the relative difference is 
        above a certain threshold.
        """

        tries, max_tries = 0, self.pop_size // self.nr_skip_parents
        other_parent = np.random.choice(sorted_controls, p=fit_norm)
        id_other = sorted_controls.index(other_parent)
        rel_distance = distances_norm[id_other]

        while rel_distance < self.min_dist_perc and tries < max_tries:
            other_parent = np.random.choice(sorted_controls, p=fit_norm)
            id_other = sorted_controls.index(other_parent)
            rel_distance = distances_norm[id_other]
            tries += 1

        return id_other, other_parent

    def make_new_generation(self, fit_norm_sorted, sorted_controls):
        """
        Crossover gense for a given population
        """

        # start creating children based on pairs of parents
        children = []
        for i in range(0, self.pop_size, self.nr_skip_parents):

            # rank selection of two parents
            _, parent1 = self.parent_selection(fit_norm_sorted, sorted_controls)
            distances_norm = self.get_distance_probs(parent1, sorted_controls)
            _, parent2 = self.select_other_parent(fit_norm_sorted, sorted_controls, distances_norm, parent1)

            # create child and add to children list
            child = self.crossover_and_mutation(parent1, parent2)
            children.append(child)

        # select parents based on normilzed probabilites of the fitnesses who
        # do not survive
        sorted_controls = self.determine_survival(fit_norm_sorted, sorted_controls, children)

        return sorted_controls

class DistanceWeightedRank(SimulationWeightedRank):
    def __init__(
            self, experiment_name, nr_inputs, nr_layers,
            nr_neurons, nr_outputs, activation_func, activation_distr,
            lower_bound, upper_bound, pop_size, nr_gens,
            mutation_chance, nr_skip_parents, enemies, multiplemode, min_dist_perc
        ):
        super().__init__(
            experiment_name, nr_inputs, nr_layers, nr_neurons,
            nr_outputs, activation_func, activation_distr,
            lower_bound, upper_bound, pop_size, nr_gens,
            mutation_chance, nr_skip_parents, enemies,  multiplemode
        )
        self.min_dist_perc = min_dist_perc

    def get_distance_probs(self, parent1, sorted_controls):
        """
        Determines the  probability based on the distance of the first
        parent selected relatively to that of all the others.
        """
        distances, params1 = [], parent1.get_params()
        for other_parent in sorted_controls:
            params2 = other_parent.get_params()
            distance = 0
            
            for layer in range(self.tot_layers):
                weight_str = "W" + str(layer)
                W1, W2 = params1[weight_str], params2[weight_str]
                diff = (W1 - W2).flatten()
                distance += np.sqrt(np.dot(diff, diff))

            distances.append(distance)

        # normalize, so we get probabilities
        # sum_distances = sum(distances)
        # distances_norm = [distance / sum_distances for distance in distances]
        min_distance, max_distance = min(distances), max(distances)
        range_distance = max_distance - min_distance
        distances_norm = [(distance - min_distance) / range_distance for distance in distances]

        return distances_norm

    def select_other_parent(self, fit_norm, sorted_controls, distances_norm, parent1):
        """
        Selects other parent and only accepts if the relative difference is 
        above a certain threshold.
        """

        tries, max_tries = 0, self.pop_size // self.nr_skip_parents
        other_parent = np.random.choice(sorted_controls, p=fit_norm)
        id_other = sorted_controls.index(other_parent)
        rel_distance = distances_norm[id_other]

        while rel_distance < self.min_dist_perc and tries < max_tries:
            other_parent = np.random.choice(sorted_controls, p=fit_norm)
            id_other = sorted_controls.index(other_parent)
            rel_distance = distances_norm[id_other]
            tries += 1

        return id_other, other_parent

    def make_new_generation(self, fit_norm_sorted, sorted_controls, fitnesses):
        """
        Crossover gense for a given population
        """

        # start creating children based on pairs of parents
        children = []
        for i in range(0, self.pop_size, self.nr_skip_parents):

            # rank selection of two parents
            id1, parent1 = self.parent_selection(fit_norm_sorted, sorted_controls)
            distances_norm = self.get_distance_probs(parent1, sorted_controls)
            id2, parent2 = self.select_other_parent(fit_norm_sorted, sorted_controls, distances_norm, parent1)

            # create child and add to children list
            child = self.crossover_and_mutation(parent1, parent2, id1, id2, fitnesses)
            children.append(child)

        # select parents based on normilzed probabilites of the fitnesses who
        # do not survive
        sorted_controls = self.determine_survival(fit_norm_sorted, sorted_controls, children)

        return sorted_controls

class DistanceRankAdaptiveNpoint(SimulationAdaptiveMutationNpointCrossover):
    def __init__(
            self, experiment_name, nr_inputs, nr_layers,
            nr_neurons, nr_outputs, activation_func, activation_distr,
            lower_bound, upper_bound, pop_size, nr_gens,
            mutation_chance, nr_skip_parents, enemies, multiplemode, min_dist_perc
        ):
        super().__init__(
            experiment_name, nr_inputs, nr_layers, nr_neurons,
            nr_outputs, activation_func, activation_distr,
            lower_bound, upper_bound, pop_size, nr_gens,
            mutation_chance, nr_skip_parents, enemies,  multiplemode
        )
        self.min_dist_perc = min_dist_perc

    def get_distance_probs(self, parent1, sorted_controls):
        """
        Determines the  probability based on the distance of the first
        parent selected relatively to that of all the others.
        """
        distances, params1 = [], parent1.get_params()
        for other_parent in sorted_controls:
            params2 = other_parent.get_params()
            distance = 0
            
            for layer in range(self.tot_layers):
                weight_str = "W" + str(layer)
                W1, W2 = params1[weight_str], params2[weight_str]
                diff = (W1 - W2).flatten()
                distance += np.sqrt(np.dot(diff, diff))

            distances.append(distance)

        # normalize, so we get probabilities
        # sum_distances = sum(distances)
        # distances_norm = [distance / sum_distances for distance in distances]
        min_distance, max_distance = min(distances), max(distances)
        range_distance = max_distance - min_distance
        distances_norm = [(distance - min_distance) / range_distance for distance in distances]

        return distances_norm

    def select_other_parent(self, fit_norm, sorted_controls, distances_norm, parent1):
        """
        Selects other parent and only accepts if the relative difference is 
        above a certain threshold.
        """

        tries, max_tries = 0, self.pop_size // self.nr_skip_parents
        other_parent = np.random.choice(sorted_controls, p=fit_norm)
        id_other = sorted_controls.index(other_parent)
        rel_distance = distances_norm[id_other]

        while rel_distance < self.min_dist_perc and tries < max_tries:
            other_parent = np.random.choice(sorted_controls, p=fit_norm)
            id_other = sorted_controls.index(other_parent)
            rel_distance = distances_norm[id_other]
            tries += 1

        return id_other, other_parent

    def make_new_generation(self, fit_norm_sorted, sorted_controls, fitnesses):
        """
        Crossover gense for a given population
        """

        # start creating children based on pairs of parents
        children = []
        for i in range(0, self.pop_size, self.nr_skip_parents):

            # rank selection of two parents
            id1, parent1 = self.parent_selection(fit_norm_sorted, sorted_controls)
            distances_norm = self.get_distance_probs(parent1, sorted_controls)
            id2, parent2 = self.select_other_parent(fit_norm_sorted, sorted_controls, distances_norm, parent1)

            # create child and add to children list
            child = self.crossover_and_mutation(parent1, parent2, id1, id2, fitnesses)
            children.append(child)

        # select parents based on normilzed probabilites of the fitnesses who
        # do not survive
        sorted_controls = self.determine_survival(fit_norm_sorted, sorted_controls, children)

        return sorted_controls