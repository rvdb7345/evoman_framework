from ga_algorithms_npoint import GA_random_Npoint

class GA_random_linear(GA_random_Npoint):
    def crossover(self, child_params, prob1, prob2, W1, W2, b1, b2, str_layer):
        """
        Linear weighted crossover of the weights in a certain layer
        """
        child_params["W" + str_layer] = prob1 * W1 + prob2 * W2
        child_params["b" + str_layer] = prob1 * b1 + prob2 * b2

        return child_params