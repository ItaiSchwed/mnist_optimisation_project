import torch

import settingsModule
from settingsModule import Settings, np, random
from population import Population



class GeneticAlgorithm:
	settings: Settings
	population: Population
	mutation_probability: float
	generation_number: int

	def __init__(self, settings: Settings):
		self.settings = settings
		self.mutation_probability = self.settings.mutation_probability
		self.population = Population(settings)
		self.generation_number = 0

	def create_new_generation(self):
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
		self.generation_number += 1

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
			if self.settings.mutation == settingsModule.MUTATIONS.UNIFORMAL_PER_GENE:
				offsprings[offspring_idx] = self.uniformal_mutation(offsprings[offspring_idx])
			elif self.settings.mutation == settingsModule.MUTATIONS.INDIVIDUAL_PER_OFFSPRING:
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


	def get_iteration_num(self):
		return self.generation_number

	def get_picture_title(self):
		title = f"GEN:{self.generation_number}, Label:{self.settings.label}\n"

		if self.settings.penalty_factor != 0:
			return title + "Certainty:{0:.2f}%".format(round(self.get_population_best()['certainty']*100, 2)) +\
				   ", Penalty:{0:.4f}".format(round(self.get_population_best()['penalty'], 4)) + \
				   ", Fitness:{0:.4f}".format(round(self.get_population_best()['fitness'], 4))
		else:
			return title + "Fitness:{0:.2f}%".format(round(self.get_population_best()['fitness']*100, 2))

	def get_progress_status(self):
		status = f"GEN:{self.generation_number}, Label:{self.settings.label} | BEST("

		if self.settings.penalty_factor != 0:
			status += "Certainty:{0:.4f}".format(round(self.get_population_best()['certainty'], 4)) + \
				   ", Penalty:{0:.4f}".format(round(self.get_population_best()['penalty'], 4)) + \
				   ", Fitness:{0:.4f}".format(round(self.get_population_best()['fitness'], 4))
		else:
			status += "Fitness:{0:.4f}".format(round(self.get_population_best()['fitness'], 4))

		status += ") | Worst(Fitness:{0:.4f})".format(round(self.get_population_wrost()['fitness'], 4))
		return status

	def termination_condition(self):
		return self.population.termination_condition()

class GreedyAlgorithm:
	settings: Settings

	def __int__(self, settings: Settings):
		self.settings = settings

		pic = np.random.random_integers(0, self.settings.num_of_shades - 1,
										(self.settings.num_of_layers, self.settings.picture_size, self.settings.picture_size)) \
			  / (self.settings.num_of_shades - 1)

		max_fitness = 0
		last_max_fitness = -1
		while last_max_fitness != max_fitness:
			last_max_fitness = max_fitness

			for i in range(pic.shape[1]):
				for j in range(pic.shape[2]):
					max_fitness_shade = pic[0][i][j]

					for shade in range(self.settings.num_of_shades):
						pic[0][i][j] = shade / (self.settings.num_of_shades - 1)
						fitness = self.certainty(pic)
						if fitness > max_fitness:
							max_fitness = fitness
							max_fitness_shade = shade / (self.settings.num_of_shades - 1)

					pic[0][i][j] = max_fitness_shade
					print(last_max_fitness==max_fitness, i, j, max_fitness)

	def certainty(self, pic: np.array):
		with torch.no_grad():
			network_input = torch.Tensor(pic).unsqueeze(dim=0)  # Expending the array (one example)
			network_output = self.settings.network(network_input)
			certainty = torch.nn.functional.softmax(network_output, dim=1)

			return certainty[0][self.settings.label_num]