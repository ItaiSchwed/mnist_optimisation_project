import os
import torch
import numpy as np
import random
from enum import Enum

from models import MnistModel, FashionMnistModel2, CatsDogsModel

# The answer for everything :)
np.random.seed(42)
random.seed(42)

# Optimization Type
class OPTIMIZATIONS(Enum):
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
	optimization: OPTIMIZATIONS
	model: MODELS
	cross_over: CROSS_OVERS
	mutation: MUTATIONS
	mutation_probability: float
	penalty_factor: float
	poputation_size: int
	num_of_shades: int
	num_of_layers: int
	picture_size: int
	label_num: int
	label: str
	network: torch.nn.Module

	RESULTS_PATH = "results/"
	TRAINED_MODELS_PATH = "trainedModels/"  # The folder of the trained models
	STATUS_PERIOD = 1000  # After which number of generations, the progress status will be printed
	DISPLAY = DISPLAYS.PICTURE_ONLY

	def __init__(self, optimization:OPTIMIZATIONS, model:MODELS, label_num:int, cross_over:CROSS_OVERS, mutation:MUTATIONS,
				 mutation_probability:float, penalty_factor:float, population_size:int, num_of_shades:int):
		self.optimization = optimization
		self.model = model
		self.label_num = label_num
		self.cross_over = cross_over
		self.mutation = mutation
		self.mutation_probability = mutation_probability
		self.penalty_factor = penalty_factor
		self.population_size = population_size
		self.num_of_shades = num_of_shades

		if not os.path.exists(self.RESULTS_PATH):
			os.makedirs(self.RESULTS_PATH)

		self.RESULTS_PATH += "/" + self.get_settings_str()

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
			network_state_dict = torch.load(self.TRAINED_MODELS_PATH + "fashion-mnist-model2", map_location=torch.device('cpu'))
			self.network.load_state_dict(network_state_dict)

			self.label = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
						  "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"][self.label_num]
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

	def get_settings_str(self):
		settings_str = "Model=" + self.model.name + "_Optimization=" + self.optimization.name + "_Label=" + self.label

		if self.optimization == OPTIMIZATIONS.GA:
			settings_str += "_CrossOver=" + self.cross_over.name + "_Mutation=" + self.mutation.name + \
							"_mutation_prob=" + str(self.mutation_probability) + "_Penalty=" + str(self.penalty_factor) + \
							"_PopulationSize=" + str(self.population_size) + "_Shades=" + str(self.num_of_shades)

		settings_str += ".png"
		return settings_str

'''
# Models Properties
class MODELS(Enum):
	MNIST = 0
	FASHION_MNIST = 1
	DOGS_CATS = 2

TRAINED_MODELS_PATH = "trainedModels/"	# The folder of the trained models
MODEL = MODELS.MNIST
LABEL_NUM = 1
LABEL = ""
POPULATION_SIZE = 50

# Defaults Values
NUM_OF_SHADES = 1						# Between 1 to 256
NUM_OF_LAYERS = 1						# Number of color types (1 for black-white, and 3 for regular RGB)
PICTURE_SIZE = 1
NETWORK: torch.nn.Module

if MODEL == MODELS.MNIST:
	NETWORK = MnistModel()
	network_state_dict = torch.load(TRAINED_MODELS_PATH + "mnist-model")
	NETWORK.load_state_dict(network_state_dict)

	LABELS = range(10)
	LABEL = str(LABELS[LABEL_NUM])
	NUM_OF_SHADES = 256
	NUM_OF_LAYERS = 1
	PICTURE_SIZE = 28

elif MODEL == MODELS.FASHION_MNIST:
	NETWORK = FashionMnistModel2()
	network_state_dict = torch.load(TRAINED_MODELS_PATH + "fashion-mnist-model2", map_location=torch.device('cpu'))
	NETWORK.load_state_dict(network_state_dict)

	LABELS = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
	LABEL = LABELS[LABEL_NUM]
	NUM_OF_SHADES = 256
	NUM_OF_LAYERS = 1
	PICTURE_SIZE = 28

elif MODEL == MODELS.DOGS_CATS:
	NETWORK = CatsDogsModel
	network_state_dict = torch.load(TRAINED_MODELS_PATH + "cats-dogs-model", map_location=torch.device('cpu'))
	NETWORK.load_state_dict(network_state_dict)

	LABELS = ["Cat", "Dog"]
	LABEL = LABELS[LABEL_NUM]
	NUM_OF_SHADES = 256
	NUM_OF_LAYERS = 3
	PICTURE_SIZE = 500

else:
	print("Illegal Model!")
	exit()


# Cross-overs Properties
class CROSS_OVERS(Enum):
	ONE_POINT = 0
	TWO_POINT = 1
	UNIFORM = 2

CROSS_OVER = CROSS_OVERS.UNIFORM


# Mutations Properties
class MUTATIONS(Enum):
	PER_GENE = 0			# Mutation for each gen in a specific probability
	INDIVIDUAL = 1			# Mutation for only one gen

MUTATION = MUTATIONS.INDIVIDUAL
OFFSPRING_MUTATION_PROPABILITY = 0.7
GENE_MUTATION_PROPABILITY = 0.01


# Penalty Properties
penalty_factor = 0.1


# Display Properties
class DISPLAYS(Enum):
	GRAPH = 0			# Graph + Picture (Slowest)
	PICTURE_ONLY = 1	# Only Picutre
	COMMAND_LINE = 2	# Information appears in command line (Fastest)

DISPLAY = DISPLAYS.PICTURE_ONLY
STATUS_PERIOD = 1000 	# After which number of generations, the progress status will be printed


# Output Pictures Path
if not os.path.exists('results'):
	os.makedirs('results')

PICTURE_PATH = "results/Model=" + MODEL.name + "_Optimization=" + OPTIMIZATION.name + "_Label=" + LABEL

if OPTIMIZATION == OPTIMIZATIONS.GA:
	PICTURE_PATH += "_CrossOver=" + CROSS_OVER.name + "_Mutation=" + MUTATION.name + \
					"_Penalty=" + str(penalty_factor) + "_PopulationSize=" + str(POPULATION_SIZE) + "Shades=" + NUM_OF_SHADES

PICTURE_PATH += ".png"
'''