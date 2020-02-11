import os
import torch
import numpy as np
import random
from enum import Enum

from models import MnistModel, FashionMnistModel, FashionMnistModel2, CatsDogsModel

# The answer for everything :)
np.random.seed(42)
random.seed(42)

# Optimization Type
class ALGORITHMS(Enum):
	GA = 0
	GREEDY = 1

# Models Properties
class MODELS(Enum):
	MNIST = 0
	FASHION_MNIST = 1
	DOGS_CATS = 2

# Cross-overs Properties
class CROSS_OVERS(Enum):
	ONE_POINT = 0
	TWO_POINT = 1
	UNIFORM = 2

# Mutations Properties
class MUTATIONS(Enum):
	UNIFORMAL_PER_GENE = 0					# Mutation for each gen in a specific probability
	INDIVIDUAL_PER_OFFSPRING = 1			# Mutation for only one gen

# Display Properties
class DISPLAYS(Enum):
	GRAPH = 0			# Graph + Picture (Slowest)
	PICTURE_ONLY = 1	# Only Picutre
	COMMAND_LINE = 2	# Information appears in command line (Fastest)


class Settings:
	algorithm_type: ALGORITHMS
	model: MODELS
	cross_over: CROSS_OVERS
	mutation: MUTATIONS
	mutation_probability: float
	penalty_factor: float
	poputation_size: int
	num_of_shades: int				# Between 2 to 256
	num_of_layers: int				# 1 layer (black-white) or 3 layers (regular RGB)
	picture_size: int
	status_period: int				# After which number of iterations, the progress status will be printed
	label_num: int
	label: str
	network: torch.nn.Module

	RESULTS_PATH = "results"
	TRAINED_MODELS_PATH = "trainedModels/"  # The folder of the trained models
	DISPLAY = DISPLAYS.PICTURE_ONLY

	def __init__(self, algorithm_type:ALGORITHMS, model:MODELS, label_num:int, cross_over:CROSS_OVERS, mutation:MUTATIONS,
				 mutation_probability:float, penalty_factor:float, population_size:int, num_of_shades:int, status_period):

		self.algorithm_type = algorithm_type
		self.model = model
		self.label_num = label_num
		self.cross_over = cross_over
		self.mutation = mutation
		self.mutation_probability = mutation_probability
		self.penalty_factor = penalty_factor
		self.population_size = population_size
		self.num_of_shades = num_of_shades
		self.status_period= status_period

		self.label = ""
		self.num_of_layers = 1
		self.picture_size = 1

		if self.model == MODELS.MNIST:
			self.network = MnistModel()
			network_state_dict = torch.load(self.TRAINED_MODELS_PATH + "mnist-model", map_location=torch.device('cpu'))
			self.network.load_state_dict(network_state_dict)

			self.label = str(range(10)[self.label_num])
			self.num_of_layers = 1
			self.picture_size = 28

		elif self.model == MODELS.FASHION_MNIST:
			self.network = FashionMnistModel2()
			network_state_dict = torch.load(self.TRAINED_MODELS_PATH + "fashion-mnist-model-2", map_location=torch.device('cpu'))
			self.network.load_state_dict(network_state_dict)

			self.label = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
						  "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"][self.label_num]
			self.num_of_layers = 1
			self.picture_size = 28

		elif self.model == MODELS.DOGS_CATS:
			self.network = CatsDogsModel
			network_state_dict = torch.load(self.TRAINED_MODELS_PATH + "cats-dogs-model", map_location=torch.device('cpu'))
			self.network.load_state_dict(network_state_dict)

			self.label = ["Cat", "Dog"][self.label_num]
			self.num_of_layers = 3
			self.picture_size = 500

		else:
			print("Illegal Model!")
			exit()

		if not os.path.exists(self.RESULTS_PATH):
			os.makedirs(self.RESULTS_PATH)

		self.RESULTS_PATH += "/" + self.get_settings_str()

	def get_settings_str(self):
		settings_str = "Model=" + self.model.name + "_Algorithm=" + self.algorithm_type.name + "_Label=" + self.label

		if self.algorithm_type == ALGORITHMS.GA:
			settings_str += "_CrossOver=" + self.cross_over.name + "_Mutation=" + self.mutation.name + \
							"_mutation_prob=" + str(self.mutation_probability) + "_Penalty=" + str(self.penalty_factor) + \
							"_PopulationSize=" + str(self.population_size) + "_Shades=" + str(self.num_of_shades)

		settings_str += ".png"
		return settings_str