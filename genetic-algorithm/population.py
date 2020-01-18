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
                           .apply(lambda row:
                                  pd.Series([
                                      self.fitness(row[0]),
                                      row[1] * 10,
                                      row[2] * 10,
                                      row[3] * 10,
                                      row[4] * 10],
                                      index=['fitness_value',
                                             'budget',
                                             'leading_actor_count',
                                             'oscar_avg_per_actor',
                                             'movie_length']),
                                  axis=1
                                  )
                           )
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

    @staticmethod
    def fitness(off_spring: pd.Series):
        return off_spring.loc['budget']


population = Population()
new_offsprings = pd.DataFrame({
    'fitness_value': [0.8, 0.6],
    'budget': [20000, 30000],
    'leading_actor_count': [2, 3],
    'oscar_avg_per_actor': [0.1, 0.2],
    'movie_length': [3000000, 300000]})
population.steady_state_replace(new_offsprings)
