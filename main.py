from matplotlib import pyplot as plt
from matplotlib import animation, image
from PIL import Image

import glob
import settingsModule
from settingsModule import Settings, np
from algorithms import Algorithm, GeneticAlgorithm, GreedyAlgorithm


# Generating the best suitable input for the network according to the given settings
class TransparentGenerator:
	settings: Settings
	algorithm: Algorithm
	last_best_fitness: float

	graph_x = []
	graph_y = []

	def __init__(self, algorithm_class, settings: Settings):
		self.settings = settings
		self.algorithm = algorithm_class(settings)
		self.last_best_fitness = -np.inf # invalid fitness value as the initial last one

		self.results_file = open(self.settings.RESULTS_PATH.replace(".png", ".txt"), "w")

		# Print in command line (standard output)
		if self.settings.DISPLAY == settingsModule.DISPLAYS.COMMAND_LINE:
			while not self.algorithm.termination_condition():
				self.algorithm.iterate()

				if self.algorithm.iteration_num % self.settings.status_period == 0:
					print(self.algorithm.get_progress_status())
					self.results_file.write(self.algorithm.get_file_progress() + "\n")
					self.results_file.flush()

				self.algorithm.iteration_num += 1

			print("STOPPED")

			fig = plt.figure()
			fig.subplots_adjust(top=0.8)
			fig.suptitle('Transparent Generator- ' + self.algorithm.get_algorithm_name(), fontsize=12, fontweight='bold')
			self.img = fig.add_subplot(1, 1, 1)
			self.img.axis("off")
			self.update_progress()

		# Show a picture over time
		elif self.settings.DISPLAY == settingsModule.DISPLAYS.PICTURE_ONLY:
			fig = plt.figure()
			fig.subplots_adjust(top=0.8)
			fig.suptitle('Transparent Generator- ' + self.algorithm.get_algorithm_name(), fontsize=12, fontweight='bold')
			self.img = fig.add_subplot(1, 1, 1)
			self.img.axis("off")
			self.animation = animation.FuncAnimation(fig, self.animate_only_picture, interval=1)
			plt.show()

		# Show a progress graph
		elif self.settings.DISPLAY == settingsModule.DISPLAYS.GRAPH:
			fig = plt.figure(figsize=(15, 7))
			fig.suptitle('Transparent Generator- ' + self.algorithm.get_algorithm_name(), fontsize=12, fontweight='bold')
			self.graph = fig.add_subplot(1, 2, 1)
			self.graph.set_xlabel("Number of Generations")
			self.graph.set_ylabel("Fitness")
			self.img = fig.add_subplot(1, 2, 2)
			self.img.axis("off")

			fig.subplots_adjust(top=0.87)
			self.line, = self.graph.plot([], [], lw=2)
			self.animation = animation.FuncAnimation(fig, self.animate_graph, init_func=self.init, interval=1, blit=True)
			plt.show()

		else:
			print("Illegal Display Option!")
			exit()

		self.results_file.close()

	def animate_only_picture(self, i):
		if self.algorithm.iteration_num != 0:
			self.algorithm.iterate()

		if self.algorithm.iteration_num % self.settings.status_period == 0:
			# Update the image
			self.update_progress()

		if self.algorithm.termination_condition():
			self.update_progress()
			self.animation.event_source.stop()
			print("STOPPED")

		self.algorithm.iteration_num += 1

	def init(self):
		self.line.set_data([], [])
		return self.line,

	def animate_graph(self, i):
		if self.algorithm.iteration_num != 0:
			self.algorithm.iterate()

		self.graph_x.append(i)
		self.graph_y.append(self.algorithm.get_best_fitness())
		self.reset_plot_bounds()

		if self.algorithm.iteration_num % self.settings.status_period == 0:
			# Updating the graph
			self.line.set_data(self.graph_x, self.graph_y)
			self.graph.figure.canvas.draw()
			self.update_progress()

		if self.algorithm.termination_condition():
			self.update_progress()
			self.animation.event_source.stop()
			print("STOPPED")

		self.algorithm.iteration_num += 1

		return self.line,

	def update_progress(self):
		print(self.algorithm.get_progress_status())

		# Write Progress in the resutls file
		self.results_file.write(self.algorithm.get_file_progress() + "\n")
		self.results_file.flush()

		# Update image only if the best fitness value was changed
		if self.algorithm.get_best_fitness() != self.last_best_fitness:
			self.img.set_title(self.algorithm.get_picture_title())
			self.last_best_fitness = self.algorithm.get_best_fitness()

			if self.settings.num_of_layers == 1:
				self.img.imshow(self.algorithm.get_best_picture()[0], cmap='gray', interpolation='none')
				image.imsave(self.settings.RESULTS_PATH, np.array(Image.fromarray(self.algorithm.get_best_picture()[0]).resize((self.settings.picture_size*4, self.settings.picture_size*4))), cmap='gray')
			else: # 3 layers
				self.img.imshow(np.swapaxes(self.algorithm.get_best_picture(), 0, 2), interpolation='none')
				image.imsave(self.settings.RESULTS_PATH, np.swapaxes(self.algorithm.get_best_picture(), 0, 2))

	def reset_plot_bounds(self):
		x_max = self.graph.get_xlim()[1]
		if len(self.graph_x) >= x_max:
			self.graph.set_xlim(0, 2 * x_max)

		y_max = self.graph.get_ylim()[1]
		if self.graph_y[len(self.graph_y) - 1] >= y_max:
			self.graph.set_ylim(0, 2 * y_max)

def run_one_algorithm(algorithm_type, model, label_num, cross_over, mutation, mutation_probability, penalty_factor, population_size, num_of_shades, status_period):
	algorithm = Algorithm

	if algorithm_type == settingsModule.ALGORITHMS.GA:
		algorithm = GeneticAlgorithm
		status_period = 1000
	elif algorithm_type == settingsModule.ALGORITHMS.GREEDY:
		algorithm = GreedyAlgorithm
		status_period = 100
	else:
		print("Illegal Algorithm!")
		exit()

	settings = Settings(algorithm_type=algorithm_type, model=model, label_num=label_num,
						cross_over=cross_over, mutation=mutation, mutation_probability=mutation_probability,
						penalty_factor=penalty_factor, population_size=population_size, num_of_shades=num_of_shades,
						status_period=status_period)

	TransparentGenerator(algorithm, settings)

def show_progress_graph(algorithm_type, model, cross_over, mutation, mutation_probability, penalty_factor, population_size, num_of_shades, status_period):
	#files = glob.glob(Settings.RESULTS_PATH + "/*.txt")
	fig = plt.figure(figsize=(23, 7))

	certainty_graph = fig.add_subplot(131)
	certainty_graph.set_title("Certainty over Time")
	certainty_graph.set_xlabel("Iteartions")
	certainty_graph.set_ylabel("Certainty")

	penalty_graph = fig.add_subplot(132)
	penalty_graph.set_title("Penalty over Time")
	penalty_graph.set_xlabel("Iteartions")
	penalty_graph.set_ylabel("Peanlty")

	fitness_graph = fig.add_subplot(133)
	fitness_graph.set_title("Fitness over Time")
	fitness_graph.set_xlabel("Iteartions")
	fitness_graph.set_ylabel("Fitness")


	for label_num in range(10):
		settings = Settings(algorithm_type=algorithm_type, model=model, label_num=label_num,
							cross_over=cross_over, mutation=mutation, mutation_probability=mutation_probability,
							penalty_factor=penalty_factor, population_size=population_size, num_of_shades=num_of_shades,
							status_period=status_period)

		label = "Digit " + str(label_num)
		iterations = []
		penalties = []
		certainties = []
		fitnesses = []

		with(open(settings.RESULTS_PATH.replace(".png", ".txt"), "r")) as f:
			for line in f.readlines():
				iteration, certainty, penalty, fitness = line.replace("\n\r", "").split(", ")

				iteration = int(iteration.split(":")[1])
				certainty = float(certainty.split(":")[1])
				penalty = float(penalty.split(":")[1])
				fitness = float(fitness.split(":")[1])

				iterations.append(iteration)
				certainties.append(certainty)
				penalties.append(penalty)
				fitnesses.append(fitness)

		certainty_graph.plot(iterations, certainties, label=label)
		penalty_graph.plot(iterations, penalties, label=label)
		fitness_graph.plot(iterations, fitnesses, label=label)

	certainty_graph.legend()
	penalty_graph.legend()
	fitness_graph.legend()
	plt.show()


def main():
	import sys

	model = settingsModule.MODELS.MNIST
	cross_over = settingsModule.CROSS_OVERS.UNIFORM
	mutation = settingsModule.MUTATIONS.UNIFORMAL
	algorithm_type = settingsModule.ALGORITHMS.GA
	mutation_probability = 0.01 # 0.7 # 0.1
	penalty_factor = 0.05 # 0
	population_size = 50
	num_of_shades = 256 # 2
	status_period = 1
	label_num = 0

	if len(sys.argv) == 2:
		if sys.argv[1] == "-all":
			for algorithm_type in settingsModule.ALGORITHMS:
				if algorithm_type == settingsModule.ALGORITHMS.GREEDY:
					exit()

				num_of_labels = 10
				if model == settingsModule.MODELS.DOGS_CATS:
					num_of_labels = 2

				for label_num in range(num_of_labels):
					run_one_algorithm(algorithm_type, model, label_num,
									  cross_over, mutation, mutation_probability,
									  penalty_factor, population_size, num_of_shades,
									  status_period)
		elif sys.argv[1] == "-graph":
			show_progress_graph(algorithm_type, model,
								cross_over, mutation, mutation_probability,
								penalty_factor, population_size, num_of_shades,
								status_period)

	else:
		run_one_algorithm(algorithm_type, model, label_num,
						  cross_over, mutation, mutation_probability,
						  penalty_factor, population_size, num_of_shades,
						  status_period)

if __name__ == "__main__":
	main()