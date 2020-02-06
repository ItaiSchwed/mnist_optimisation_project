import numpy as np
import random
from enum import Enum

# The answer for everything :)
np.random.seed(42)
random.seed(42)


# Models Properties
class MODELS(Enum):
	FASHION_MNIST = 0
	MNIST = 1

TRAINED_MODELS_PATH = "trainedModels/"	# The folder of the trained models
MODEL = MODELS.MNIST
LABEL_NUM = 0							# Between 0 to 9
NUM_OF_COLORS = 256						# Between 1 to 256
PICTURE_SIZE = 28						# ONLY 28
POPULATION_SIZE = 50					# Can be any integer

if MODEL == MODELS.MNIST:
	MODEL_NAME = TRAINED_MODELS_PATH + "mnist-model"
	LABELS = range(10)
	LABEL = LABELS[LABEL_NUM]
elif MODEL == MODELS.FASHION_MNIST:
	MODEL_NAME = TRAINED_MODELS_PATH + "fashion-mnist-model"
	LABELS = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
	LABEL = LABELS[LABEL_NUM]
else:
	print("Illegal Model!")
	exit()


# Cross-overs Properties
class CROSS_OVERS(Enum):
	ONE_POINT = 0
	TWO_POINT = 1
	UNIFORM = 2

CROSS_OVER = CROSS_OVERS.TWO_POINT


# Mutations Properties
class MUTATIONS(Enum):
	PER_GEN = 0			# Mutation for each gen in a specific probability
	INDIVIDUAL = 1		# Mutation for only one gen

MUTATION = MUTATIONS.INDIVIDUAL
OFFSPRING_MUTATION_PROPABILITY = 0.7
GENE_MUTATION_PROPABILITY = 0.001


# Penalty Properties
WITH_PENALTY = False
PENALTY_FACTOR = 0.5


# Display Properties
class DISPLAYS(Enum):
	GRAPH = 0			# Graph + Picture (Slowest)
	PICTURE_ONLY = 1	# Only Picutre
	COMMAND_LINE = 2	# Information appears in command line (Fastest)

DISPLAY = DISPLAYS.GRAPH
STATUS_PERIOD = 1000 	# After which number of generations, the progress status will be printed