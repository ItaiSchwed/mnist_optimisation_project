from matplotlib import pyplot as plt
from matplotlib import animation

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

		# Print in command line (standard output)
		if self.settings.DISPLAY == settingsModule.DISPLAYS.COMMAND_LINE:
			while not self.algorithm.termination_condition():
				self.algorithm.iterate()

				if self.algorithm.iteration_num % self.settings.status_period == 0:
					print(self.algorithm.get_progress_status())

				self.algorithm.iteration_num += 1

			fig = plt.figure()
			fig.subplots_adjust(top=0.8)
			fig.suptitle('Transparent Generator- ' + self.algorithm.get_algorithm_name(), fontsize=12, fontweight='bold')
			img = fig.add_subplot(1, 1, 1)
			img.axis("off")
			img.title(self.algorithm.get_picture_title())

			if settings.num_of_layers == 1:
				img.imshow(self.algorithm.get_best_picture()[0], cmap='gray', interpolation='none')
			else:  # 3 layers
				img.imshow(np.swapaxes(self.algorithm.get_best_picture(), 0, 2), interpolation='none')

			plt.savefig(self.settings.RESULTS_PATH)

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

	def animate_only_picture(self, i):
		if self.algorithm.iteration_num != 0:
			self.algorithm.iterate()

		if self.algorithm.iteration_num % self.settings.status_period == 0:
			print(self.algorithm.get_progress_status())

			if self.algorithm.get_best_fitness() != self.last_best_fitness:
				self.img.set_title(self.algorithm.get_picture_title())
				self.last_best_fitness = self.algorithm.get_best_fitness()

				if self.settings.num_of_layers == 1:
					self.img.imshow(self.algorithm.get_best_picture()[0], cmap='gray', interpolation='none')
				else: # 3 layers
					self.img.imshow(np.swapaxes(self.algorithm.get_best_picture(), 0, 2), interpolation='none')

				plt.savefig(self.settings.RESULTS_PATH)

			if self.algorithm.termination_condition():
				self.animation.event_source.stop()

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
			print(self.algorithm.get_progress_status())

			self.line.set_data(self.graph_x, self.graph_y)
			self.graph.figure.canvas.draw()

			# Update image only if the best fitness was changed
			if self.algorithm.get_best_fitness() != self.last_best_fitness:
				self.img.set_title(self.algorithm.get_picture_title())
				self.last_best_fitness = self.algorithm.get_best_fitness()

				if self.settings.num_of_layers == 1:
					self.img.imshow(self.algorithm.get_best_picture()[0], cmap='gray', interpolation='none')
				else:  # 3 layers
					self.img.imshow(np.swapaxes(self.algorithm.get_best_picture(), 0, 2), interpolation='none')

				plt.savefig(self.settings.RESULTS_PATH)

			if self.algorithm.termination_condition():
				self.animation.event_source.stop()

		self.algorithm.iteration_num += 1

		return self.line,

	def reset_plot_bounds(self):
		x_max = self.graph.get_xlim()[1]
		if len(self.graph_x) >= x_max:
			self.graph.set_xlim(0, 2 * x_max)

		y_max = self.graph.get_ylim()[1]
		if self.graph_y[len(self.graph_y) - 1] >= y_max:
			self.graph.set_ylim(0, 2 * y_max)

def main():
	import sys

	model = settingsModule.MODELS.MNIST
	cross_over = settingsModule.CROSS_OVERS.UNIFORM
	mutation = settingsModule.MUTATIONS.INDIVIDUAL_PER_OFFSPRING
	mutation_probability = 0.7  # 0.01
	penalty_factor = 0.1
	population_size = 50
	num_of_shades = 256 # 2
	algorithm = Algorithm
	status_period = 1

	if len(sys.argv) == 2 and sys.argv[1] == "-all":
		for algorithm_type in settingsModule.ALGORITHMS:
			if algorithm_type == settingsModule.ALGORITHMS.GA:
				algorithm = GeneticAlgorithm
				status_period = 1000
			elif algorithm_type == settingsModule.ALGORITHMS.GREEDY:
				algorithm = GreedyAlgorithm
				status_period = 1
			else:
				print("Illegal Algorithm!")
				exit()

			num_of_labels = 10
			if model == settingsModule.MODELS.DOGS_CATS:
				num_of_labels = 2

			for label_num in range(num_of_labels):
				settings = Settings(algorithm_type=algorithm_type, model=model, label_num=label_num,
									cross_over=cross_over, mutation=mutation, mutation_probability=mutation_probability,
									penalty_factor=penalty_factor, population_size=population_size, num_of_shades=num_of_shades,
									status_period=status_period)

				TransparentGenerator(algorithm, settings)

	else:
		algorithm_type = settingsModule.ALGORITHMS.GA
		label_num = 0

		if algorithm_type == settingsModule.ALGORITHMS.GA:
			algorithm = GeneticAlgorithm
			status_period = 1000
		elif algorithm_type == settingsModule.ALGORITHMS.GREEDY:
			algorithm = GreedyAlgorithm
			status_period = 1
		else:
			print("Illegal Algorithm!")
			exit()

		settings = Settings(algorithm_type=algorithm_type, model=model, label_num=label_num,
							cross_over=cross_over, mutation=mutation, mutation_probability=mutation_probability,
							penalty_factor=penalty_factor, population_size=population_size, num_of_shades=num_of_shades,
							status_period=status_period)

		TransparentGenerator(algorithm, settings)

if __name__ == "__main__":
	main()