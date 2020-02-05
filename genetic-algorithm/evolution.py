import random
import pandas as pd
import numpy as np

from models.enums import Action
from models.rubik_state import RubikState
from population import Population


class Evolution:
    population: Population
    mutation_probability: float

    def __init__(self, startingState: RubikState):
        self.population = Population(startingState)
        self.mutation_probability = 0.1

    def create_new_generation(self):
        new_offsprings = self.uniform_crossover()
        self.population.steady_state_replace(self.random_resetting_mutations(new_offsprings))

    def uniform_crossover(self):
        parent1, parent2 = self.population.roulette_wheel_select_2_parents()
        offsprings = pd.DataFrame({}, columns=Population.get_population_columns_names())
        for column, _ in offsprings.iteritems():
            random_boolean = bool(random.getrandbits(1))
            offsprings.loc[0, column] = parent1[column] if random_boolean else parent2[column]
            offsprings.loc[1, column] = parent2[column] if random_boolean else parent1[column]
        # offsprings.loc[0, :25] = parent1[:25]
        # offsprings.loc[0, 25:] = parent2[25:]
        # offsprings.loc[1, :25] = parent2[:25]
        # offsprings.loc[1, 25:] = parent1[25:]
        for index, offspring in offsprings.iterrows():
            offsprings.loc[index, 'fitness_value'] = self.population.fitness(offspring)
        return offsprings

    def random_resetting_mutations(self, offsprings: pd.DataFrame):
        for index, offspring in offsprings.iterrows():
            if random.random() < self.mutation_probability:
                offsprings.loc[index, :] = self.__perform_mutation(offspring)
        return offsprings

    def get_population_max_fitness(self):
        return self.population.get_population_max_fitness()

    def get_population_min_fitness(self):
        return self.population.get_population_min_fitness()

    def __perform_mutation(self,offspring: pd.Series):
        for column, _ in offspring.iteritems():
            offspring[column] = random.choice([action for action in Action])
        # gen_indexes = np.random.randint(0, offspring.count() - 1, random.randint(0, offspring.count())
        # self.swap(gen_indexes, offspring)
        offspring['fitness_value'] = self.population.fitness(offspring)
        return offspring

    def swap(self, gen_indexes, offspring):
        temp = offspring.loc[gen_indexes[0]]
        offspring.loc[gen_indexes[0]] = offspring.loc[gen_indexes[1]]
        offspring.loc[gen_indexes[1]] = temp

    def termination_condition(self):
        return self.population.termination_condition()
