import random

import pandas as pd
import numpy as np

from models.enums import Action
from models.rubik_state import RubikState


class Population:
    __population: pd.DataFrame
    __evolution_counter: int
    starting_state: RubikState

    def __init__(self, starting_state: RubikState):
        self.starting_state = starting_state
        self.__random_initialize_population()
        self.__evolution_counter = 0

    @staticmethod
    def get_population_columns_names():
        return list(range(50))

    def __random_initialize_population(self):
        self.population = pd.DataFrame(np.random.choice([action for action in Action], (100, 50)))
        for index, row in self.population.iterrows():
            self.population.loc[index, 'fitness_value'] = self.fitness(row)
        return

    def fitness(self, offspring: pd.Series):
        state: RubikState = RubikState(self.starting_state.state)
        for i in range(50):
            state.action(offspring[i])
        return state.getSolvingPercent()

    def steady_state_replace(self, new_offsprings: pd.DataFrame):
        new_offsprings = new_offsprings.sort_values('fitness_value').assign(new=True)
        population_2_smallest = (self.population.nsmallest(2, 'fitness_value')
                                 .reset_index())
        indexes = population_2_smallest.loc[:, 'index'].values
        largest_2_offsprings = (population_2_smallest.assign(new=False)
                                .append(new_offsprings,
                                        sort=False)
                                .nlargest(2, 'fitness_value').set_index(keys=indexes))
        columns = list(range(50))
        columns.append('fitness_value')
        for index in indexes:
            self.population.loc[index] = largest_2_offsprings.loc[index].loc[columns]
        if largest_2_offsprings.apply(lambda row: row.loc['new'], axis=1).any():
            self.__evolution_counter = 0
        else:
            self.__evolution_counter += 1

    def get_population_max_fitness(self):
        return self.population.nlargest(1, 'fitness_value').iloc[0]

    def get_population_min_fitness(self):
        return self.population.nsmallest(1, 'fitness_value').iloc[0]

    def roulette_wheel_select_2_parents(self):
        sum_of_finesses = (self.population
                           .apply(lambda chromosome: chromosome['fitness_value'], axis=1)
                           .sum())
        return (self.__roulette_wheel_select_parent(sum_of_finesses),
                self.__roulette_wheel_select_parent(sum_of_finesses))

    def __roulette_wheel_select_parent(self, sum_of_finesses):
        random_number = random.uniform(0, sum_of_finesses)
        for index, chromosome in self.population.iterrows():
            random_number -= chromosome['fitness_value']
            if random_number <= 0:
                return chromosome
        raise ValueError('Internal error')

    def termination_condition(self):
        return False
        # return self.__evolution_counter >= 100
