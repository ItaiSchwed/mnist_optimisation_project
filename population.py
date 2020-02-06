import torch

import settings
from settings import np
from models import MnistModel, FashionMnistModel


class Population:
	__evolution_counter: int

	population_pics: np.array
	population_fitnesses: np.array
	population_certainties: np.array
	population_penalties: np.array
	network: torch.nn.Module
	penalty_factor: float
	population_size: float
	with_penalty: bool

	def __init__(self):
		if settings.MODEL == settings.MODELS.MNIST:
			self.network = MnistModel()
		elif settings.MODEL == settings.MODELS.FASHION_MNIST:
			self.network = FashionMnistModel()
		else:
			print("Illegal Model!")
			exit()

		network_state_dict = torch.load(settings.MODEL_NAME)
		self.network.load_state_dict(network_state_dict)
		self.network.eval()

		self.population_size = settings.POPULATION_SIZE
		self.penalty_factor = settings.PENALTY_FACTOR
		self.with_penalty = settings.WITH_PENALTY
		self.__random_initialize_population()
		self.__evolution_counter = 0

	def __random_initialize_population(self):
		self.population_pics = np.random.random_integers(0, settings.NUM_OF_COLORS-1, (self.population_size, settings.PICTURE_SIZE, settings.PICTURE_SIZE)) / (settings.NUM_OF_COLORS-1)
		self.population_fitnesses = np.random.random(self.population_size)
		self.population_certainties = np.random.random(self.population_size)
		self.population_penalties = np.random.random(self.population_size)

		for picture_idx in range(len(self.population_pics)):
			certainty = self.certainty(self.population_pics[picture_idx])
			penalty = self.penalty(self.population_pics[picture_idx])

			self.population_certainties[picture_idx] = certainty
			self.population_penalties[picture_idx] = penalty
			self.population_fitnesses[picture_idx] = self.fitness(certainty, penalty)


	def certainty(self, offspring: np.array):
		with torch.no_grad():
			network_input = torch.Tensor(offspring).unsqueeze(dim=0).unsqueeze(dim=0) # Expending the array (one example, one kernel)
			network_output = self.network(network_input)

			return network_output[0][settings.LABEL_NUM]

	def penalty(self, offspring: np.array):
		# The penalty will be the distances between every two adjacent pixels in the picture
		penalty = 0

		# Avoiding penalty computation when possible
		if self.with_penalty:
			row_neighbors = np.copy(offspring)
			row_neighbors[1:, :] = offspring[:-1, :]

			col_neighbors = np.copy(offspring)
			col_neighbors[:, 1:] = offspring[:, :-1]

			#penalty = np.sum(np.sqrt(np.abs(offspring - row_neighbors)) + np.sqrt(np.abs(offspring - col_neighbors)))
			penalty = np.sum(np.abs(offspring - row_neighbors) + np.abs(offspring - col_neighbors))
			penalty /= pow(settings.PICTURE_SIZE, 2)

		return penalty

	def fitness(self, certainty, penalty):
		if self.with_penalty:
			fitness = certainty - self.penalty_factor*penalty
		else:
			fitness = certainty

		return fitness


	def steady_state_replace(self, offsprings_pics: np.array):
		first_offspring_changed = self.steady_state_replace_one_offspring(offsprings_pics[0])
		second_offspring_changed = self.steady_state_replace_one_offspring(offsprings_pics[1])

		if not first_offspring_changed and not second_offspring_changed:
			self.__evolution_counter += 1
		else:
			self.__evolution_counter = 0

	def steady_state_replace_one_offspring(self, offspring_pic: np.array):
		min_fitness_idx = np.argmin(self.population_fitnesses)

		offspring_certainty = self.certainty(offspring_pic)
		offspring_penalty = self.penalty(offspring_pic)
		offspring_fitness = self.fitness(offspring_certainty, offspring_penalty)

		if offspring_fitness < self.population_fitnesses[min_fitness_idx]:
			return False
		else:
			self.population_pics[min_fitness_idx] = offspring_pic
			self.population_certainties[min_fitness_idx] = offspring_certainty
			self.population_penalties[min_fitness_idx] = offspring_penalty
			self.population_fitnesses[min_fitness_idx] = offspring_fitness
			return True


	def get_population_max_fitness(self):
		return np.max(self.population_fitnesses)

	def get_population_min_fitness(self):
		return np.min(self.population_fitnesses)

	def get_population_best_pic(self):
		return self.population_pics[np.argmax(self.population_fitnesses)]

	def get_certainty_of_max_fitness(self):
		return self.population_certainties[np.argmax(self.population_fitnesses)]

	def get_penalty_of_max_fitness(self):
		return self.population_penalties[np.argmax(self.population_fitnesses)]


	def roulette_wheel_select_2_parents(self):
		sum_of_fitnesses = np.sum(np.abs(self.population_fitnesses))
		relative_fitness = np.abs(self.population_fitnesses) / sum_of_fitnesses
		selected_parents_indexes = np.random.choice(range(len(self.population_pics)), p=relative_fitness, size=2, replace=False)
		return self.population_pics[selected_parents_indexes[0]], self.population_pics[selected_parents_indexes[1]]


	def termination_condition(self):
		return self.__evolution_counter >= 100