import random
import pandas as pd

from population import Population


class Evolution:
    population: Population
    mutation_probability: float

    def __init__(self):
        self.population = Population()
        self.mutation_probability = 0.1

    def create_new_generation(self):
        new_offsprings = self.uniform_crossover()
        self.population.steady_state_replace(self.random_resetting_mutations(new_offsprings))
        return self.population.get_population_min_fitness()

    def uniform_crossover(self):
        parent1, parent2 = self.population.roulette_wheel_select_2_parents()
        offsprings = pd.DataFrame({}, columns=Population.get_population_columns_names())
        for column in Population.get_population_columns_names():
            random_bool = bool(random.getrandbits(1))
            offsprings.loc[0, column] = parent1[column] if random_bool else parent2[column]
            offsprings.loc[1, column] = parent2[column] if random_bool else parent1[column]
        for index, offspring in offsprings.iterrows():
            offsprings.loc[index, 'fitness_value'] = Population.fitness(offspring)
        return offsprings

    def random_resetting_mutations(self, offsprings: pd.DataFrame):
        for index, offspring in offsprings.iterrows():
            if random.random() < self.mutation_probability:
                offsprings.loc[index, :] = Evolution.__perform_mutation(offspring)
        return offsprings

    @staticmethod
    def __perform_mutation(offspring: pd.Series):
        gen_index = random.randint(0, offspring.count() - 1)
        mutation = random.uniform(0, 10)
        offspring[gen_index] = mutation
        return offspring

    def termination_condition(self):
        return self.population.termination_condition()
