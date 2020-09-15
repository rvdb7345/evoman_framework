from simulation import SimulationRank
from simulation import SimulationRoulette
from simulation import SimulationWeightedRank
from simulation import SimulationAdaptiveMutationRank
from simulation import SimulationAdaptiveMutationRoulette
from simulation import SimulationAdaptiveMutationWeightedRank
from simulation import SimulationAdaptiveMutationNpointCrossover
from simulation import SimulationScrambledMutation
from simulation import SimulationSwapMutation

from heuristics import DistanceRank, DistanceRoulette, DistanceWeightedRank
from heuristics import DistanceRankAdaptiveNpoint


if __name__ == "__main__":
<<<<<<< HEAD
    
    # sim = SimulationRank(
    #     experiment_name="basic_sim_rank",
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
    #     multiplemode="no"
    # )
    # sim.run_evolutionary_algo()
    # print("FINISHED!!!!")
=======
    sim = SimulationRank(
        experiment_name="basic_sim_rank",
        nr_inputs=20,
        nr_layers=1,
        nr_neurons=10,
        nr_outputs=5,
        activation_func=["sigmoid"],
        activation_distr=[1],
        lower_bound=-1,
        upper_bound=1,
        pop_size=10,
        nr_gens=5,
        mutation_chance=0.2,
        nr_skip_parents=2,
        enemies=[8],
        multiplemode="no"
    )
    sim.run_evolutionary_algo()
    print("FINISHED!!!!")
>>>>>>> 490a14d6c372e2a0f40680934e30cffcfd59320a
    
    sim = SimulationRoulette(
        experiment_name="basic_sim_roulette",
        nr_inputs=20,
        nr_layers=1,
        nr_neurons=10,
        nr_outputs=5,
        activation_func=["sigmoid"],
        activation_distr=[1],
        lower_bound=-1,
        upper_bound=1,
        pop_size=10,
        nr_gens=5,
        mutation_chance=0.2,
        nr_skip_parents=2,
        enemies=[8],
        multiplemode="no"
    )
    sim.run_evolutionary_algo()
    print("FINISHED!!!!")

    sim = SimulationWeightedRank(
        experiment_name="basic_weighted_rank",
        nr_inputs=20,
        nr_layers=1,
        nr_neurons=10,
        nr_outputs=5,
        activation_func=["sigmoid"],
        activation_distr=[1],
        lower_bound=-1,
        upper_bound=1,
        pop_size=10,
        nr_gens=5,
        mutation_chance=0.2,
        nr_skip_parents=2,
        enemies=[8],
        multiplemode="no"
    )
    sim.run_evolutionary_algo()
    print("FINISHED!!!!")

<<<<<<< HEAD
    

    #sim = DistanceRoulette(
    #    experiment_name="basic_distance_roulette",
    #    nr_inputs=20,
    #    nr_layers=1,
    #    nr_neurons=10,
    #    nr_outputs=5,
    #    activation_func=["sigmoid"],
    #    activation_distr=[1],
    #    lower_bound=-1,
    #    upper_bound=1,
    #    pop_size=10,
    #    nr_gens=5,
    #    mutation_chance=0.2,
    #    nr_skip_parents=2,
    #    enemies=[8],
    #    multiplemode="no",
    #    min_dist_perc=0.1
    #)
    #sim.run_evolutionary_algo()
    #print("FINISHED!!!!")

    #sim = DistanceWeightedRank(
    #    experiment_name="basic_distance_weightedRank",
    #    nr_inputs=20,
    #    nr_layers=1,
    #    nr_neurons=10,
    #    nr_outputs=5,
    #    activation_func=["sigmoid"],
    #    activation_distr=[1],
    #    lower_bound=-1,
    #  upper_bound=1,
    #    pop_size=10,
    #   nr_gens=5,
    #    mutation_chance=0.2,
    #    nr_skip_parents=2,
    #    enemies=[8],
    #    multiplemode="no",
    #    min_dist_perc=0.1
    #)
    #sim.run_evolutionary_algo()
    #print("FINISHED!!!!")

    #sim = DistanceRankAdaptiveNpoint(
    #    experiment_name="basic_distance_adaptiveRankNpoint",
    #    nr_inputs=20,
    #    nr_layers=1,
    #    nr_neurons=10,
    #    nr_outputs=5,
    #    activation_func=["sigmoid"],
    #    activation_distr=[1],
    #    lower_bound=-1,
    #    upper_bound=1,
    #    pop_size=10,
    #    nr_gens=5,
    #    mutation_chance=0.2,
    #    nr_skip_parents=2,
    #    enemies=[8],
    #    multiplemode="no",
    #    min_dist_perc=0.1
    #)
    #sim.run_evolutionary_algo()
    
    #print("FINISHED!!!!")
    
    
    #sim = SimulationAdaptiveMutationNpointCrossover(
    #     experiment_name="weighted_sim_npoint",
    #     nr_inputs=20,
    #     nr_layers=1,
    #     nr_neurons=10,
    #     nr_outputs=5,
    #     activation_func=["sigmoid"],
    #     activation_distr=[1],
    #     lower_bound=-1,
    #     upper_bound=1,
    #     pop_size=10,
    #    nr_gens=5,
    #    mutation_chance=0.2,
    #     nr_skip_parents=2,
    #     enemies=[8],
    #     multiplemode="no"
    # )
    
    #sim.run_evolutionary_algo()
    #print("FINISHED!!!!")
=======
    sim = SimulationAdaptiveMutationNpointCrossover(
        experiment_name="weighted_sim_npoint",
        nr_inputs=20,
        nr_layers=1,
        nr_neurons=10,
        nr_outputs=5,
        activation_func=["sigmoid"],
        activation_distr=[1],
        lower_bound=-1,
        upper_bound=1,
        pop_size=10,
        nr_gens=5,
        mutation_chance=0.2,
        nr_skip_parents=2,
        enemies=[8],
        multiplemode="no"
    )
    sim.run_evolutionary_algo()
    print("FINISHED!!!!")

    sim = DistanceRank(
        experiment_name="basic_distance_rank",
        nr_inputs=20,
        nr_layers=1,
        nr_neurons=10,
        nr_outputs=5,
        activation_func=["sigmoid"],
        activation_distr=[1],
        lower_bound=-1,
        upper_bound=1,
        pop_size=10,
        nr_gens=5,
        mutation_chance=0.2,
        nr_skip_parents=2,
        enemies=[8],
        multiplemode="no",
        min_dist_perc=0.1
    )
    sim.run_evolutionary_algo()
    print("FINISHED!!!!")
>>>>>>> 490a14d6c372e2a0f40680934e30cffcfd59320a

    #sim = SimulationScrambledMutation(
    #     experiment_name="Scrambled_mutation",
    #     nr_inputs=20,
    #     nr_layers=1,
    #     nr_neurons=10,
    #     nr_outputs=5,
    #     activation_func=["sigmoid"],
    #     activation_distr=[1],
    #     lower_bound=-1,
    #     upper_bound=1,
    #     pop_size=10,
    #    nr_gens=5,
    #    mutation_chance=0.2,
    #     nr_skip_parents=2,
    #     enemies=[8],
    #     multiplemode="no"
    # )
    
    #sim.run_evolutionary_algo()
    #print("FINISHED!!!!")

    sim = SimulationSwapMutation(
         experiment_name="swapping_mutation",
         nr_inputs=20,
         nr_layers=1,
         nr_neurons=10,
         nr_outputs=5,
         activation_func=["sigmoid"],
         activation_distr=[1],
         lower_bound=-1,
         upper_bound=1,
         pop_size=10,
        nr_gens=5,
        mutation_chance=0.2,
         nr_skip_parents=2,
         enemies=[8],
         multiplemode="no"
     )
    
    sim.run_evolutionary_algo()
    print("FINISHED!!!!")
