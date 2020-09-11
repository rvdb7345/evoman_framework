from controller import Controller
import numpy as np

class test_controller(Controller):
    """
    Implements controller structure for player
    """
    def __init__(self, nn_topology, mutation_step_size=0.1, lb=-1, ub=1):
        """
        Initializes with a list of amount of neurons. Each index
        indicates the a hidden layer
        """
        self.nn_topology = nn_topology
        self.n_layers = len(self.nn_topology)
        self.lower_bound = lb
        self.upper_bound = ub
        self.mutation_step_size = mutation_step_size

        self.params_value = {}
        self.activations_funcs = {"sigmoid": self.sigmoid_activation}

    def initialize_random_network(self, seed=None):
        """
        Initializes the weights of the neural network. Note, that a bias term is
        included.
        """
        if seed is not None:
            np.random.seed(seed)

        for idx, layer in enumerate(self.nn_topology):
            input_size = layer["input_dim"]
            output_size = layer["output_dim"]
            activation = layer["activation"]

            self.params_value['W' + str(idx)] = np.random.uniform(
                self.lower_bound, self.upper_bound, (output_size, input_size)
            )
            self.params_value['b' + str(idx)] = np.random.uniform(
                self.lower_bound, self.upper_bound, (output_size, )
            )
            self.params_value["activation" + str(idx)] = activation

    def set_mutation_step_size(self, mut_size):
        self.mutation_step_size = mut_size

    def create_network(self, params_value):
        """
        Creates/updates a neural network from a dictionary
        """
        self.params_value = params_value

    def get_params(self):
        """
        Returns parameters neural network
        """
        return self.params_value

    def sigmoid_activation(self, X):
        """
        Sigmoid activation function for neural network
        """
        return 1. / (1. + np.exp(-X))

    # def one_layer_forward(self, X_prev, W_curr, b_curr, activation="sigmoid"):
    #     """
    #     Performs the calculations in current layer.
    #     Returns output of current layer
    #     """
    #     assert activation in self.activations_funcs, "Activation function not found"

    #     V_curr = np.dot(W_curr, X_prev) + b_curr
    #     return self.activations_funcs[activation](V_curr)

    def control(self, inputs, controller):
        """
        Perfroms a full forward propagation of a neural network. Returns
        a list of values representing if the nodes in the output layer are
        activated or not (resp. 1 and 0)
        """
        inputs = (inputs - min(inputs)) / float((max(inputs) - min(inputs)))
        X_curr = inputs

        for id_layer in range(self.n_layers):
            X_prev = X_curr

            str_id = str(id_layer)
            W_curr = self.params_value['W' + str_id]
            b_curr = self.params_value['b' + str_id]
            activation = self.params_value["activation" +  str_id]

            assert activation in self.activations_funcs, "Activation function not found"

            V_curr = np.dot(W_curr, X_prev) + b_curr
            X_curr = self.activations_funcs[activation](V_curr)
            # X_curr = self.one_layer_forward(X_prev, W_curr, b_curr, activation=activation)

        # make decision for sprite action
        left = 1 if X_curr[0] > 0.5 else 0
        right = 1 if X_curr[1] > 0.5 else 0
        jump = 1 if X_curr[2] > 0.5 else 0
        shoot = 1 if X_curr[3] > 0.5 else 0
        release = 1 if X_curr[4] > 0.5 else 0

        return [left, right, jump, shoot, release]
