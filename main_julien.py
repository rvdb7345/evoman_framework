from simulation import SimulationRank
from simulation import SimulationRoulette
from simulation import SimulationWeightedRank
from simulation import SimulationAdaptiveMutationRank
from simulation import SimulationAdaptiveMutationRoulette
from simulation import SimulationAdaptiveMutationWeightedRank
from simulation import SimulationAdaptiveMutationNpointCrossover

from heuristics import DistanceRank, DistanceRoulette, DistanceWeightedRank
from heuristics import DistanceRankAdaptiveNpoint

if __name__ == "__main__":
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

    sim = DistanceRoulette(
        experiment_name="basic_distance_roulette",
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

    sim = DistanceWeightedRank(
        experiment_name="basic_distance_weightedRank",
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

    sim = DistanceRankAdaptiveNpoint(
        experiment_name="basic_distance_adaptiveRankNpoint",
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