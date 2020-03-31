import torch

import settingsModule
from settingsModule import Settings, np, random
from population import Population

# Parent of all algorithms
class Algorithm:
	iteration_num: int
	settings: Settings

	def __init__(self, settings: Settings):
		self.settings = settings
		self.iteration_num = 0

	def iterate(self) -> None:
		pass

	def get_algorithm_name(self) -> str:
		pass

	def get_picture_title(self) -> str:
		pass

	def get_progress_status(self) -> str:
		pass

	def get_file_progress(self) -> str:
		pass

	def get_best_picture(self) -> np.array:
		pass

	def get_best_fitness(self) -> float:
		pass

	def termination_condition(self) -> bool:
		pass


# Genetic Algorithm
class GeneticAlgorithm(Algorithm):
	population: Population
	mutation_probability: float

	def __init__(self, settings: Settings):
		super().__init__(settings)
		self.mutation_probability = self.settings.mutation_probability
		self.population = Population(settings)

	def iterate(self):
		if self.settings.cross_over == settingsModule.CROSS_OVERS.ONE_POINT:
			offsprings = self.one_point_crossover()
		elif self.settings.cross_over == settingsModule.CROSS_OVERS.TWO_POINT:
			offsprings = self.two_points_crossover()
		elif self.settings.cross_over == settingsModule.CROSS_OVERS.UNIFORM:
			offsprings = self.uniform_crossover()
		else:
			print("Illegal Cross Over!")
			exit()

		self.population.steady_state_replace(self.random_mutations(offsprings))

	def one_point_crossover(self):
		parent1, parent2 = self.population.roulette_wheel_select_2_parents()

		rand_col = random.randint(0, parent1.shape[1] - 1)

		offspring1 = np.copy(parent1)
		offspring1[:, :, :rand_col] = parent2[:, :, :rand_col]

		offspring2 = np.copy(parent2)
		offspring2[:, :, rand_col:] = parent1[:, :, rand_col:]

		return np.array([offspring1, offspring2])

	def two_points_crossover(self):
		parent1, parent2 = self.population.roulette_wheel_select_2_parents()

		rand_row1 = random.randint(0, parent1.shape[0] - 1)
		rand_col1 = random.randint(0, parent1.shape[1] - 1)

		rand_row2 = random.randint(0, parent1.shape[0] - 1)
		rand_col2 = random.randint(0, parent1.shape[1] - 1)

		offspring1 = np.copy(parent1)
		offspring1[:, rand_row1:rand_row2, rand_col1:rand_col2] = parent2[:, rand_row1:rand_row2, rand_col1:rand_col2]

		offspring2 = np.copy(parent2)
		offspring2[:, rand_row1:rand_row2, rand_col1:rand_col2] = parent1[:, rand_row1:rand_row2, rand_col1:rand_col2]

		return np.array([offspring1, offspring2])

	def uniform_crossover(self):
		parent1, parent2 = self.population.roulette_wheel_select_2_parents()

		random_matching = np.random.random_integers(0, 1, (parent1.shape[1], parent1.shape[2]))

		offspring1 = random_matching * parent1[:] + (1-random_matching) * parent2[:]
		offspring2 = (1-random_matching) * parent1[:] + random_matching * parent2[:]

		return np.array([offspring1, offspring2])


	def random_mutations(self, offsprings: np.array):
		for offspring_idx in range(len(offsprings)):
			if self.settings.mutation == settingsModule.MUTATIONS.UNIFORMAL:
				offsprings[offspring_idx] = self.uniformal_mutation(offsprings[offspring_idx])
			elif self.settings.mutation == settingsModule.MUTATIONS.INDIVIDUAL:
				offsprings[offspring_idx] = self.individual_mutation(offsprings[offspring_idx])
			else:
				print("Illegal Mutation!")
				exit()

		return offsprings

	def uniformal_mutation(self, offspring: np.array):
		# Make a mutation for each gene in a specific probability
		mutations_activations = np.random.rand(offspring.shape[0], offspring.shape[1], offspring.shape[2]) < self.mutation_probability
		return mutations_activations * np.random.rand(offspring.shape[0], offspring.shape[1], offspring.shape[2]) \
			   + (1 - mutations_activations) * offspring

	def individual_mutation(self, offspring: np.array):
		# Make a mutation in a specific probability
		if random.random() < self.mutation_probability:
			# Change the value in all the layers of the picture
			for i in range(self.settings.num_of_layers):
				gen_index = np.random.random_integers(0, offspring.shape[1] - 1, 2)
				mutation = random.randint(0, self.settings.num_of_shades-1) / (self.settings.num_of_shades-1)
				offspring[:, gen_index[0], gen_index[1]] = mutation

		return offspring


	def get_population_best(self):
		return self.population.get_population_best()

	def get_population_wrost(self):
		return self.population.get_population_wrost()


	def get_algorithm_name(self):
		return "Genetic Algorithm"

	def get_picture_title(self):
		title = f"Generation:{self.iteration_num}, Label:{self.settings.label}\n"

		if self.settings.penalty_factor != 0:
			return title + "Certainty:{0:.2f}%".format(round(self.get_population_best()['certainty']*100, 2)) +\
				   ", Penalty:{0:.4f}".format(round(self.get_population_best()['penalty'], 4)) + \
				   ", Fitness:{0:.4f}".format(round(self.get_population_best()['fitness'], 4))
		else:
			return title + "Fitness:{0:.2f}%".format(round(self.get_population_best()['fitness']*100, 2))

	def get_progress_status(self):
		status = f"Algorithm:GA, Model:{self.settings.model.name}, Generation:{self.iteration_num}, Label:{self.settings.label} | BEST("

		if self.settings.penalty_factor != 0:
			status += "Certainty:{0:.4f}".format(round(self.get_population_best()['certainty'], 4)) + \
				   ", Penalty:{0:.4f}".format(round(self.get_population_best()['penalty'], 4)) + \
				   ", Fitness:{0:.4f}".format(round(self.get_population_best()['fitness'], 4))
		else:
			status += "Fitness:{0:.4f}".format(round(self.get_population_best()['fitness'], 4))

		status += ") | Worst(Fitness:{0:.4f})".format(round(self.get_population_wrost()['fitness'], 4))
		return status

	def get_file_progress(self):
		return f"Iteration:{self.iteration_num}" + \
			   ", Certainty:{0:.4f}".format(round(self.get_population_best()['certainty'], 4)) + \
			   ", Penalty:{0:.4f}".format(round(self.get_population_best()['penalty'], 4)) + \
			   ", Fitness:{0:.4f}".format(round(self.get_population_best()['fitness'], 4))

	def get_best_picture(self):
		return self.get_population_best()['pic']

	def get_best_fitness(self):
		return self.get_population_best()['fitness']

	def termination_condition(self):
		return self.population.get_stuck_evolution_counter() >= 10 * self.settings.status_period or self.get_population_best()['fitness'] >= 0.99


# Greedy Algorithm
class GreedyAlgorithm(Algorithm):
	pic: np.array
	max_fitness: float
	last_max_fitness: float
	curr_i = 0
	curr_j = 0

	def __init__(self, settings: Settings):
		super().__init__(settings)
		self.pic = np.random.random_integers(0, self.settings.num_of_shades - 1,
										(self.settings.num_of_layers, self.settings.picture_size, self.settings.picture_size)) \
				   / (self.settings.num_of_shades - 1)
		self.max_fitness = self.certainty(self.pic)
		self.last_max_fitness = 0.0
		self.penalty_factor = settings.penalty_factor

	def iterate(self):
		if self.curr_i >= self.pic.shape[1]:
			self.curr_i = 0

		if self.curr_j >= self.pic.shape[2]:
			self.curr_j = 0

		max_fitness_shade = self.pic[0][self.curr_i][self.curr_j]

		for shade in range(self.settings.num_of_shades):
			#print(shade)
			self.pic[0][self.curr_i][self.curr_j] = shade / (self.settings.num_of_shades - 1)
			fitness = self.fitness(self.pic)
			if fitness > self.max_fitness:
				self.max_fitness = fitness
				max_fitness_shade = shade / (self.settings.num_of_shades - 1)

		self.pic[0][self.curr_i][self.curr_j] = max_fitness_shade

		self.curr_i += 1
		self.curr_j += 1

	def certainty(self, pic: np.array):
		with torch.no_grad():
			network_input = torch.Tensor(pic).unsqueeze(dim=0)  # Expending the array (one example)
			network_output = self.settings.network(network_input)
			certainty = torch.nn.functional.softmax(network_output, dim=1)

			return float(certainty[0][self.settings.label_num])

	def penalty(self, pic: np.array):
		# The penalty is defined by the distances between every two adjacent pixels in the picture
		penalty = 0

		# Avoiding penalty computation when possible
		if self.penalty_factor != 0:
			row_neighbors = np.copy(pic)
			row_neighbors[:, 1:, :] = pic[:, :-1, :]

			col_neighbors = np.copy(pic)
			col_neighbors[:, :, 1:] = pic[:, :, :-1]

			# penalty = np.sum(np.sqrt(np.abs(offspring - row_neighbors)) + np.sqrt(np.abs(offspring - col_neighbors)))
			penalty = np.sum(np.abs(pic - row_neighbors) + np.abs(pic - col_neighbors))
			penalty /= pow(self.settings.picture_size, 2)

		return penalty

	def fitness(self, pic):
		if self.penalty_factor != 0:
			certainty = self.certainty(pic)
			penalty = self.penalty(pic)

			fitness = (1-self.penalty_factor)*certainty + self.penalty_factor*(1-penalty)
		else:
			fitness = self.certainty(pic)

		return fitness

	def get_algorithm_name(self):
		return "Greedy Algorithm"

	def get_picture_title(self):
		title = f"Iteration:{self.iteration_num}, Label:{self.settings.label}\n"
		return title + "Fitness:{0:.2f}%".format(round(self.max_fitness*100, 2))

	def get_progress_status(self):
		status = f"Algorithm:Greedy, Model:{self.settings.model.name}, Iteration:{self.iteration_num}, Label:{self.settings.label}" + \
				 ", Fitness:{0:.4f}".format(round(self.max_fitness, 4))
		return status

	def get_file_progress(self):
		return f"Iteration:{self.iteration_num}" + \
			   ", Certainty:{0:.4f}".format(round(self.max_fitness, 4))

	def get_best_picture(self):
		return self.pic

	def get_best_fitness(self):
		return self.max_fitness

	def termination_condition(self):
		return self.last_max_fitness == self.max_fitness or self.max_fitness >= 0.99