import ga_algorithms_npoint as GA_algos

class GA_random_linear(GA_algos.GA_random_Npoint):
    def crossover(self, child_params, prob1, prob2, W1, W2, b1, b2, str_layer):
        """
        Linear weighted crossover of the weights in a certain layer
        """
        child_params["W" + str_layer] = prob1 * W1 + prob2 * W2
        child_params["b" + str_layer] = prob1 * b1 + prob2 * b2

        return child_params

class GA_roulette_randomlinear(GA_algos.GA_roulette_randomNpoint):
    def crossover(self, child_params, prob1, prob2, W1, W2, b1, b2, str_layer):
        """
        Linear weighted crossover of the weights in a certain layer
        """
        child_params["W" + str_layer] = prob1 * W1 + prob2 * W2
        child_params["b" + str_layer] = prob1 * b1 + prob2 * b2

        return child_params

class GA_roulette_randomlinear_scramblemutation(GA_algos.GA_roulette_randomNpoint_scramblemutation):
    def crossover(self, child_params, prob1, prob2, W1, W2, b1, b2, str_layer):
        """
        Linear weighted crossover of the weights in a certain layer
        """
        child_params["W" + str_layer] = prob1 * W1 + prob2 * W2
        child_params["b" + str_layer] = prob1 * b1 + prob2 * b2

        return child_params

class GA_roulette_randomlinear_adaptmutation(GA_algos.GA_roulette_randomNpoint_adaptmutation):
    def crossover(self, child_params, prob1, prob2, W1, W2, b1, b2, str_layer):
        """
        Linear weighted crossover of the weights in a certain layer
        """
        child_params["W" + str_layer] = prob1 * W1 + prob2 * W2
        child_params["b" + str_layer] = prob1 * b1 + prob2 * b2

        return child_params

class GA_roulette_weightedlinear(GA_algos.GA_roulette_weightedNpoint):
    def crossover(self, child_params, prob1, prob2, W1, W2, b1, b2, str_layer):
        """
        Linear weighted crossover of the weights in a certain layer
        """
        child_params["W" + str_layer] = prob1 * W1 + prob2 * W2
        child_params["b" + str_layer] = prob1 * b1 + prob2 * b2

        return child_params

class GA_roulette_weightedlinear_scrambledmutation(GA_algos.GA_roulette_weightedNpoint_scramblemutation):
    def crossover(self, child_params, prob1, prob2, W1, W2, b1, b2, str_layer):
        """
        Linear weighted crossover of the weights in a certain layer
        """
        child_params["W" + str_layer] = prob1 * W1 + prob2 * W2
        child_params["b" + str_layer] = prob1 * b1 + prob2 * b2

        return child_params

class GA_roulette_weightedlinear_adaptmutation(GA_algos.GA_roulette_weightedNpoint_adaptmutation):
    def crossover(self, child_params, prob1, prob2, W1, W2, b1, b2, str_layer):
        """
        Linear weighted crossover of the weights in a certain layer
        """
        child_params["W" + str_layer] = prob1 * W1 + prob2 * W2
        child_params["b" + str_layer] = prob1 * b1 + prob2 * b2

        return child_params

class GA_roulette_weightedlinear_adaptscramblemutation(GA_algos.GA_roulette_weightedNpoint_adaptscramblemutation):
    def crossover(self, child_params, prob1, prob2, W1, W2, b1, b2, str_layer):
        """
        Linear weighted crossover of the weights in a certain layer
        """
        child_params["W" + str_layer] = prob1 * W1 + prob2 * W2
        child_params["b" + str_layer] = prob1 * b1 + prob2 * b2

        return child_params

class GA_distanceroulette_randomlinear(GA_algos.GA_distanceroulette_randomNpoint):
    def crossover(self, child_params, prob1, prob2, W1, W2, b1, b2, str_layer):
        """
        Linear weighted crossover of the weights in a certain layer
        """
        child_params["W" + str_layer] = prob1 * W1 + prob2 * W2
        child_params["b" + str_layer] = prob1 * b1 + prob2 * b2

        return child_params

class GA_distanceroulette_randomlinear_scramblemutation(GA_algos.GA_distanceroulette_randomNpoint_scramblemutation):
    def crossover(self, child_params, prob1, prob2, W1, W2, b1, b2, str_layer):
        """
        Linear weighted crossover of the weights in a certain layer
        """
        child_params["W" + str_layer] = prob1 * W1 + prob2 * W2
        child_params["b" + str_layer] = prob1 * b1 + prob2 * b2

        return child_params

class GA_distanceroulette_randomlinear_adaptmutation(GA_algos.GA_distanceroulette_randomNpoint_adaptmutation):
    def crossover(self, child_params, prob1, prob2, W1, W2, b1, b2, str_layer):
        """
        Linear weighted crossover of the weights in a certain layer
        """
        child_params["W" + str_layer] = prob1 * W1 + prob2 * W2
        child_params["b" + str_layer] = prob1 * b1 + prob2 * b2

        return child_params

class GA_distanceroulette_weightedlinear(GA_algos.GA_distanceroulette_weightedNpoint):
    def crossover(self, child_params, prob1, prob2, W1, W2, b1, b2, str_layer):
        """
        Linear weighted crossover of the weights in a certain layer
        """
        child_params["W" + str_layer] = prob1 * W1 + prob2 * W2
        child_params["b" + str_layer] = prob1 * b1 + prob2 * b2

        return child_params

class GA_distanceroulette_weightedlinear_scramblemutation(GA_algos.GA_distanceroulette_weightedNpoint_scramblemutation):
    def crossover(self, child_params, prob1, prob2, W1, W2, b1, b2, str_layer):
        """
        Linear weighted crossover of the weights in a certain layer
        """
        child_params["W" + str_layer] = prob1 * W1 + prob2 * W2
        child_params["b" + str_layer] = prob1 * b1 + prob2 * b2

        return child_params

class GA_distanceroulette_weightedlinear_adaptmutation(GA_algos.GA_distanceroulette_weightedNpoint_adaptmutation):
    def crossover(self, child_params, prob1, prob2, W1, W2, b1, b2, str_layer):
        """
        Linear weighted crossover of the weights in a certain layer
        """
        child_params["W" + str_layer] = prob1 * W1 + prob2 * W2
        child_params["b" + str_layer] = prob1 * b1 + prob2 * b2

        return child_params

class GA_distanecroulette_weightedlinear_adaptscramblemutation(GA_algos.GA_distanceroulette_weightedNpoint_adaptscramblemutation):
    def crossover(self, child_params, prob1, prob2, W1, W2, b1, b2, str_layer):
        """
        Linear weighted crossover of the weights in a certain layer
        """
        child_params["W" + str_layer] = prob1 * W1 + prob2 * W2
        child_params["b" + str_layer] = prob1 * b1 + prob2 * b2

        return child_params