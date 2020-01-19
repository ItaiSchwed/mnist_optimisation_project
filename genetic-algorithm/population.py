import random

import pandas as pd
import numpy as np


class Population:
    population: pd.DataFrame
    involving_counter: int

    def __init__(self):
        self.random_initialize_population()
        self.involving_counter = 0

    def random_initialize_population(self):
        self.population = (pd.DataFrame(np.random.random((50, 5)))
                           .apply(lambda chromosome:
                                  pd.Series([
                                      chromosome[0],
                                      chromosome[1] * 10,
                                      chromosome[2] * 10,
                                      chromosome[3] * 10,
                                      chromosome[4] * 10],
                                      index=['fitness_value',
                                             'budget',
                                             'leading_actor_count',
                                             'oscar_avg_per_actor',
                                             'movie_length']),
                                  axis=1
                                  )
                           )
        for index, row in self.population.iterrows():
            self.population.loc[index, 'fitness_value'] = self.fitness(row)
        return

    def steady_state_replace(self, new_offsprings: pd.DataFrame):
        new_offsprings = new_offsprings.sort_values('fitness_value').assign(new=True)
        population_2_smallest = (self.population.nsmallest(2, 'fitness_value')
                                 .reset_index())
        indexes = population_2_smallest.loc[:, 'index'].values
        largest_2_offsprings = (new_offsprings
                                .append(population_2_smallest[
                                            ['fitness_value',
                                             'budget',
                                             'leading_actor_count',
                                             'oscar_avg_per_actor',
                                             'movie_length']].assign(new=False))
                                .nlargest(2, 'fitness_value').set_index(keys=indexes))
        for index in indexes:
            self.population.loc[index] = largest_2_offsprings.loc[index].loc[['fitness_value',
                                                                              'budget',
                                                                              'leading_actor_count',
                                                                              'oscar_avg_per_actor',
                                                                              'movie_length']]
        if largest_2_offsprings.apply(lambda row: row.loc['new'], axis=1).any():
            self.involving_counter = 0
        else:
            self.involving_counter += 1

    def choose_2_parents(self):
        sum_of_finesses = (self.population
                           .apply(lambda chromosome: chromosome['fitness_value'], axis=1)
                           .sum())
        return (self.__roulette_wheel_select_parent(sum_of_finesses),
                self.__roulette_wheel_select_parent(sum_of_finesses))

    def __roulette_wheel_select_parent(self, sum_of_finesses):
        random_number = random.uniform(0, sum_of_finesses)
        a = 7
        for index, chromosome in self.population.iterrows():
            random_number -= chromosome['fitness_value']
            if random_number <= 0:
                return chromosome
        raise ValueError('Internal error')

    @staticmethod
    def fitness(off_spring: pd.Series):
        return 1 / off_spring.loc['budget']


population = Population()
# new_offsprings = pd.DataFrame({
#     'fitness_value': [1/8, 1/6],
#     'budget': [8, 6],
#     'leading_actor_count': [2, 3],
#     'oscar_avg_per_actor': [0.1, 0.2],
#     'movie_length': [3000000, 300000]})
# population.steady_state_replace(new_offsprings)
a = population.choose_2_parents()
b = 7
