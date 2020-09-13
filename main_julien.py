from simulation import SimulationRank
from simulation import SimulationRoulette

if __name__ == "__main__":
    # sim = SimulationRank(
    #     experiment_name="test_sim",
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

    sim = SimulationRoulette(
        experiment_name="test_sim",
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