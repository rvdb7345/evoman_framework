import ga_algorithms_npoint as GA_Npoint
import ga_algorithms_linear as GA_linear

if __name__ == "__main__":

    repeats = 10

    # sim = GA_Npoint.GA_random_Npoint(
    #     experiment_name="random_npoint", 
    #     nr_inputs=20, 
    #     nr_layers=1, 
    #     nr_neurons=10, 
    #     nr_outputs=5,
    #     activation_func=["sigmoid"], 
    #     activation_distr=[1],
    #     lower_bound=-1, 
    #     upper_bound=1, 
    #     pop_size=10, 
    #     nr_gens=5, 
    #     mutation_chance=0.2, 
    #     nr_skip_parents=2,
    #     enemies=[8], 
    #     multiplemode = "no",
    #     replacement = False,
    #     show_plot=True,
    #     save_output=False
    # )
    # for i in range(repeats):
    #   sim.run_evolutionary_algo()
    # print("FINISHED RANDOM NPOINT")

    # for i in range(repeats):
    #     sim = GA_Npoint.GA_distanceroulette_weightedNpoint_adaptmutation(
    #         experiment_name="distanceroulette_weightedNpoint_adaptmutation",
    #         nr_inputs=20,
    #         nr_layers=1,
    #         nr_neurons=10,
    #         nr_outputs=5,
    #         activation_func=["sigmoid"],
    #         activation_distr=[1],
    #         lower_bound=-1,
    #         upper_bound=1,
    #         pop_size=100,
    #         nr_gens=120,
    #         mutation_chance=0.4,
    #         nr_skip_parents=4,
    #         enemies=[8],
    #         multiplemode = "no",
    #         replacement = False,
    #         show_plot=True,
    #         save_output=True
    #     )
    #
    #     sim.run_evolutionary_algo()
    #
    # print("FINISHED DISTANCE ROULETTE WEIGHTED NPOINT ADAPTIVE MUTATION")

    for i in range(repeats):
        sim = GA_Npoint.GA_roulette_weightedNpoint_adaptmutation(
            experiment_name="roulette_weightedNpoint_adaptmutation",
            nr_inputs=20,
            nr_layers=1,
            nr_neurons=10,
            nr_outputs=5,
            activation_func=["sigmoid"],
            activation_distr=[1],
            lower_bound=-1,
            upper_bound=1,
            pop_size=100,
            nr_gens=120,
            mutation_chance=0.4,
            nr_skip_parents=4,
            enemies=[8],
            multiplemode = "no",
            replacement = False,
            show_plot=True,
            save_output=True
        )
        sim.run_evolutionary_algo()
    print("FINISHED ROULETTE WEIGHTED NPOINT ADAPTIVE MUTATION")

    # sim = GA_Npoint.GA_random_linear(
    #     experiment_name="random_linear", 
    #     nr_inputs=20, 
    #     nr_layers=1, 
    #     nr_neurons=10, 
    #     nr_outputs=5,
    #     activation_func=["sigmoid"], 
    #     activation_distr=[1],
    #     lower_bound=-1, 
    #     upper_bound=1, 
    #     pop_size=10, 
    #     nr_gens=5, 
    #     mutation_chance=0.2, 
    #     nr_skip_parents=2,
    #     enemies=[8], 
    #     multiplemode = "no",
    #     replacement = False,
    #     show_plot=True,
    #     save_output=False
    # )
    # sim.run_evolutionary_algo()
    # print("FINISHED RANDOM LINEAR")

    # sim = GA_Npoint.GA_roulette_randomNpoint(
    #     experiment_name="roulette_randomNpoint", 
    #     nr_inputs=20, 
    #     nr_layers=1, 
    #     nr_neurons=10, 
    #     nr_outputs=5,
    #     activation_func=["sigmoid"], 
    #     activation_distr=[1],
    #     lower_bound=-1, 
    #     upper_bound=1, 
    #     pop_size=10, 
    #     nr_gens=5, 
    #     mutation_chance=0.2, 
    #     nr_skip_parents=2,
    #     enemies=[8], 
    #     multiplemode = "no",
    #     replacement = False,
    #     show_plot=True,
    #     save_output=False
    # )
    # sim.run_evolutionary_algo()
    # print("FINISHED ROULETTE RANDOM NPOINT")

    # sim = GA_Npoint.GA_roulette_weightedNpoint(
    #     experiment_name="roulette_weightedNpoint", 
    #     nr_inputs=20, 
    #     nr_layers=1, 
    #     nr_neurons=10, 
    #     nr_outputs=5,
    #     activation_func=["sigmoid"], 
    #     activation_distr=[1],
    #     lower_bound=-1, 
    #     upper_bound=1, 
    #     pop_size=10, 
    #     nr_gens=5, 
    #     mutation_chance=0.2, 
    #     nr_skip_parents=2,
    #     enemies=[8], 
    #     multiplemode = "no",
    #     replacement = False,
    #     show_plot=True,
    #     save_output=False
    # )
    # sim.run_evolutionary_algo()
    # print("FINISHED ROULETTE WEIGHTED NPOINT")