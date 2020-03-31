import torch

from settingsModule import Settings, np


# The Population of the GA
class Population:
	settings: Settings
	population_pics: np.array
	population_fitnesses: np.array
	population_certainties: np.array
	population_penalties: np.array
	network: torch.nn.Module
	penalty_factor: float
	population_size: float
	stuck_evolution_counter: int

	def __init__(self, settings:Settings):
		self.settings = settings
		self.network = settings.network
		self.network.eval()

		self.population_size = settings.population_size
		self.penalty_factor = settings.penalty_factor
		self.random_initialize_population()
		self.stuck_evolution_counter = 0

	def random_initialize_population(self):
		self.population_pics = np.random.random_integers(0, self.settings.num_of_shades - 1,
														 (self.population_size, self.settings.num_of_layers,
														  self.settings.picture_size, self.settings.picture_size)) \
							   / (self.settings.num_of_shades-1)

		# Black-White when the picture has 3 layers
		#for picture_idx in range(int(self.population_size)):
		#	self.population_pics[picture_idx, 1, :, :] = self.population_pics[picture_idx, 0, :, :]
		#	self.population_pics[picture_idx, 2, :, :] = self.population_pics[picture_idx, 0, :, :]

		# Starting with exact same color for all the picture in the same image
		#for picture_idx in range(int(self.population_size)):
		#	self.population_pics[picture_idx, :, :, :] = self.population_pics[picture_idx, :, 0, 0]

		self.population_fitnesses = np.random.random(self.population_size)
		self.population_certainties = np.random.random(self.population_size)
		self.population_penalties = np.random.random(self.population_size)

		for picture_idx in range(len(self.population_pics)):
			penalty = self.penalty(self.population_pics[picture_idx])
			certainty = self.certainty(self.population_pics[picture_idx])

			self.population_certainties[picture_idx] = certainty
			self.population_penalties[picture_idx] = penalty
			self.population_fitnesses[picture_idx] = self.fitness(certainty, penalty)


	def certainty(self, offspring: np.array):
		with torch.no_grad():
			network_input = torch.Tensor(offspring).unsqueeze(dim=0) # Expending the array (one example)
			network_output = self.network(network_input)
			certainty = torch.nn.functional.softmax(network_output, dim=1)

			return certainty[0][self.settings.label_num]

	def penalty(self, offspring: np.array):
		# The penalty is defined by the distances between every two adjacent pixels in the picture
		penalty = 0

		#offspring[0] = (offspring[0] > 0.5) + (offspring[0] > 0.5) * (offspring[0] - 1)

		# Avoiding penalty computation when possible
		if self.penalty_factor != 0:
			row_neighbors = np.copy(offspring)
			row_neighbors[:, 1:, :] = offspring[:, :-1, :]

			col_neighbors = np.copy(offspring)
			col_neighbors[:, :, 1:] = offspring[:, :, :-1]

			#penalty = np.sum(np.sqrt(np.abs(offspring - row_neighbors)) + np.sqrt(np.abs(offspring - col_neighbors)))
			penalty = np.sum(np.abs(offspring - row_neighbors) + np.abs(offspring - col_neighbors))
			penalty /= pow(self.settings.picture_size, 2)

			#penalty += np.sum(offspring) / pow(self.settings.picture_size, 2)

			# Trying other penalties
			'''
			true_color = (offspring < 0.5)[0]

			for i in range(true_color.shape[0] - 2):
				for j in range(true_color.shape[1] - 2):
					curr = true_color[i + 1][j + 1]
					if not(curr == true_color[i + 1][j] and curr == true_color[i][j + 1] and curr == true_color[i + 2][j + 1] and curr == true_color[i + 1][j + 2]):
						penalty += 1

			penalty /= pow(self.settings.picture_size, 2)
			'''

		return penalty

	def fitness(self, certainty, penalty):
		if self.penalty_factor != 0:
			fitness = (1-self.penalty_factor)*certainty + self.penalty_factor*(1-penalty)
		else:
			fitness = certainty

		return fitness


	def steady_state_replace(self, offsprings_pics: np.array):
		first_offspring_changed = self.steady_state_replace_one_offspring(offsprings_pics[0])
		second_offspring_changed = self.steady_state_replace_one_offspring(offsprings_pics[1])

		if not first_offspring_changed and not second_offspring_changed:
			self.stuck_evolution_counter += 1
		else:
			self.stuck_evolution_counter = 0

	def steady_state_replace_one_offspring(self, offspring_pic: np.array):
		min_fitness_idx = np.argmin(self.population_fitnesses)
		max_fitness_idx = np.argmax(self.population_fitnesses)

		offspring_penalty = self.penalty(offspring_pic)
		offspring_certainty = self.certainty(offspring_pic)
		offspring_fitness = self.fitness(offspring_certainty, offspring_penalty)

		found_better_offspring = False

		if round(float(offspring_fitness), 4) > round(self.population_fitnesses[max_fitness_idx], 4):
			found_better_offspring = True

		if offspring_fitness >= self.population_fitnesses[min_fitness_idx]:
			self.population_pics[min_fitness_idx] = offspring_pic
			self.population_certainties[min_fitness_idx] = offspring_certainty
			self.population_penalties[min_fitness_idx] = offspring_penalty
			self.population_fitnesses[min_fitness_idx] = offspring_fitness

		return found_better_offspring


	def get_population_best(self):
		best_fitness_idx = np.argmax(self.population_fitnesses)
		return {'fitness': np.max(self.population_fitnesses),
				'certainty': self.population_certainties[best_fitness_idx],
				'penalty': self.population_penalties[best_fitness_idx],
				'pic': self.population_pics[best_fitness_idx]}

	def get_population_wrost(self):
		wrost_fitness_idx = np.argmin(self.population_fitnesses)
		return {'fitness': np.min(self.population_fitnesses),
				'certainty': self.population_certainties[wrost_fitness_idx],
				'penalty': self.population_penalties[wrost_fitness_idx],
				'pic': self.population_pics[wrost_fitness_idx]}

	def get_stuck_evolution_counter(self):
		return self.stuck_evolution_counter

	def roulette_wheel_select_2_parents(self):
		sum_of_fitnesses = np.sum(np.abs(self.population_fitnesses))
		relative_fitness = np.abs(self.population_fitnesses) / sum_of_fitnesses
		selected_parents_indexes = np.random.choice(range(len(self.population_pics)), p=relative_fitness, size=2, replace=False)
		return self.population_pics[selected_parents_indexes[0]], self.population_pics[selected_parents_indexes[1]]