import settings
from settings import np, random
from population import Population


class Evolution:
	population: Population
	offspring_mutation_probability: float
	gene_mutation_probability: float

	def __init__(self):
		self.population = Population()
		self.offspring_mutation_probability = settings.OFFSPRING_MUTATION_PROPABILITY
		self.gene_mutation_probability = settings.GENE_MUTATION_PROPABILITY

	def create_new_generation(self):
		if settings.CROSS_OVER == settings.CROSS_OVERS.ONE_POINT:
			offsprings = self.one_point_crossover()
		elif settings.CROSS_OVER == settings.CROSS_OVERS.TWO_POINT:
			offsprings = self.two_points_crossover()
		elif settings.CROSS_OVER == settings.CROSS_OVERS.UNIFORM:
			offsprings = self.uniform_crossover()
		else:
			print("Illegal Cross Over!")
			exit()

		self.population.steady_state_replace(self.random_mutations(offsprings))

	def one_point_crossover(self):
		parent1, parent2 = self.population.roulette_wheel_select_2_parents()

		rand_col = random.randint(0, parent1.shape[1] - 1)

		offspring1 = np.copy(parent1)
		offspring1[:, :rand_col] = parent2[:, :rand_col]

		offspring2 = np.copy(parent2)
		offspring2[:, rand_col:] = parent1[:, rand_col:]

		return np.array([offspring1, offspring2])

	def two_points_crossover(self):
		parent1, parent2 = self.population.roulette_wheel_select_2_parents()

		rand_row1 = random.randint(0, parent1.shape[0] - 1)
		rand_col1 = random.randint(0, parent1.shape[1] - 1)

		rand_row2 = random.randint(0, parent1.shape[0] - 1)
		rand_col2 = random.randint(0, parent1.shape[1] - 1)

		offspring1 = np.copy(parent1)
		offspring1[rand_row1:rand_row2, rand_col1:rand_col2] = parent2[rand_row1:rand_row2, rand_col1:rand_col2]

		offspring2 = np.copy(parent2)
		offspring2[rand_row1:rand_row2, rand_col1:rand_col2] = parent1[rand_row1:rand_row2, rand_col1:rand_col2]

		return np.array([offspring1, offspring2])

	def uniform_crossover(self):
		parent1, parent2 = self.population.roulette_wheel_select_2_parents()

		random_matching = np.random.random_integers(0, 1, (parent1.shape[0], parent1.shape[1]))

		offspring1 = random_matching * parent1 + (1-random_matching) * parent2
		offspring2 = (1-random_matching) * parent1 + random_matching * parent2

		return np.array([offspring1, offspring2])


	def random_mutations(self, offsprings: np.array):
		for offspring_idx in range(len(offsprings)):
			if random.random() < self.offspring_mutation_probability:
				if settings.MUTATION == settings.MUTATIONS.PER_GEN:
					mutations_activations = np.random.rand(offsprings.shape[1], offsprings.shape[2]) < self.gene_mutation_probability
					offsprings[offspring_idx] = mutations_activations * np.random.rand(offsprings.shape[1], offsprings.shape[2]) + (1 - mutations_activations) * offsprings[offspring_idx]
				elif settings.MUTATION == settings.MUTATIONS.INDIVIDUAL:
					offsprings[offspring_idx] = Evolution.__perform_mutation(offsprings[offspring_idx])
				else:
					print("Illegal Mutation!")
					exit()

		return offsprings

	@staticmethod
	def __perform_mutation(offspring: np.array):
		gen_index = np.random.random_integers(0, offspring.shape[0] - 1, 2)
		mutation = random.randint(0, settings.NUM_OF_COLORS-1) / (settings.NUM_OF_COLORS-1)
		offspring[gen_index[0], gen_index[1]] = mutation

		return offspring


	def get_population_max_fitness(self):
		return self.population.get_population_max_fitness()

	def get_population_min_fitness(self):
		return self.population.get_population_min_fitness()

	def get_population_best_pic(self):
		return self.population.get_population_best_pic()

	def get_certainty_of_max_fitness(self):
		return self.population.get_certainty_of_max_fitness()

	def get_penalty_of_max_fitness(self):
		return self.population.get_penalty_of_max_fitness()


	def termination_condition(self):
		return self.population.termination_condition()