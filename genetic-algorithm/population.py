import random

import pandas as pd
import numpy as np


class Population:
    __population: pd.DataFrame
    __evolution_counter: int

    def __init__(self):
        self.__random_initialize_population()
        self.__evolution_counter = 0

    @staticmethod
    def get_population_columns_names():
        return ['budget',
                'leading_actor_count',
                'oscar_avg_per_actor',
                'movie_length']

    def __random_initialize_population(self):
        self.population = (pd.DataFrame(np.random.random((50, 5)))
                           .apply(lambda chromosome:
                                  pd.Series([
                                      chromosome[0],
                                      chromosome[1],
                                      chromosome[2],
                                      chromosome[3],
                                      chromosome[4]],
                                      index=['fitness_value',
                                             'budget',
                                             'leading_actor_count',
                                             'oscar_avg_per_actor',
                                             'movie_length']),
                                  axis=1
                                  )
                           )
        for index, row in self.population.iterrows():
            self.population.loc[index, 'fitness_value'] = Population.fitness(row)
        return

    @staticmethod
    def fitness(offspring: pd.Series):
        x = 5
        y = 9
        return y / (pow((offspring[Population.get_population_columns_names()].mean() - x), 2) + 1)

    def steady_state_replace(self, new_offsprings: pd.DataFrame):
        new_offsprings = new_offsprings.sort_values('fitness_value').assign(new=True)
        population_2_smallest = (self.population.nsmallest(2, 'fitness_value')
                                 .reset_index())
        indexes = population_2_smallest.loc[:, 'index'].values
        largest_2_offsprings = (population_2_smallest[
                                    ['fitness_value',
                                     'budget',
                                     'leading_actor_count',
                                     'oscar_avg_per_actor',
                                     'movie_length']].assign(new=False)
                                .append(new_offsprings,
                                        sort=False)
                                .nlargest(2, 'fitness_value').set_index(keys=indexes))
        for index in indexes:
            self.population.loc[index] = largest_2_offsprings.loc[index].loc[['fitness_value',
                                                                              'budget',
                                                                              'leading_actor_count',
                                                                              'oscar_avg_per_actor',
                                                                              'movie_length']]
        if largest_2_offsprings.apply(lambda row: row.loc['new'], axis=1).any():
            self.__evolution_counter = 0
        else:
            self.__evolution_counter += 1

    def get_population_max_fitness(self):
        return self.population.nlargest(1, 'fitness_value').iloc[0]

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
        return self.__evolution_counter >= 100
