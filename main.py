import torch
from matplotlib import pyplot as plt
from matplotlib import animation

import settingsModule
from evolution import Evolution
from settingsModule import Settings, np


class GeneticAlgorithm:
	settings: Settings
	evolution: Evolution
	last_max_fitness: float
	generation_number: int

	graph_x = []
	graph_y = []

	def __init__(self, settings: Settings):
		self.settings = settings
		self.evolution = Evolution(settings)
		self.evolution.create_new_generation()
		self.last_max_fitness = 0.0
		self.generation_number = 0

		# Show a picture over time
		if self.settings.DISPLAY == settingsModule.DISPLAYS.PICTURE_ONLY:
			fig = plt.figure()
			fig.subplots_adjust(top=0.8)
			fig.suptitle('Genetic Algorithm- Transparent Generator', fontsize=12, fontweight='bold')
			self.img = fig.add_subplot(1, 1, 1)
			self.img.axis("off")
			self.animation = animation.FuncAnimation(fig, self.animate_only_picture, interval=1)
			plt.show()

		# Print in command line (standard output)
		elif self.settings.DISPLAY == settingsModule.DISPLAYS.COMMAND_LINE:
			while not self.evolution.termination_condition():
				self.evolution.create_new_generation()

				self.generation_number += 1
				if self.generation_number % self.settings.STATUS_PERIOD == 0:
					print(self.get_progress_status())

					if self.evolution.get_population_best()['certainty'] > 0.99:
						if settings.num_of_layers == 1:
							self.img.imshow(self.evolution.get_population_best()['pic'][0], cmap='gray',
											interpolation='none')
						else:
							self.img.imshow(self.evolution.get_population_best()['pic'], interpolation='none')
						plt.show()

		# Show a progress graph
		elif self.settings.DISPLAY == settingsModule.DISPLAYS.GRAPH:
			fig = plt.figure(figsize=(15, 7))
			fig.suptitle('MNIST- Genetic Algorithm', fontsize=12, fontweight='bold')
			self.graph = fig.add_subplot(1, 2, 1)
			self.graph.set_xlabel("Number of Cross-Overs")
			self.graph.set_ylabel("Fitness")
			self.img = fig.add_subplot(1, 2, 2)
			self.img.axis("off")

			fig.subplots_adjust(top=0.87)
			self.line, = self.graph.plot([], [], lw=2)

			self.animation = animation.FuncAnimation(fig, self.animate_graph, init_func=self.init, interval=1,
													 blit=True)
			plt.show()

		else:
			print("Illegal Display Option!")
			exit()

	def animate_only_picture(self, i):
		self.evolution.create_new_generation()

		if self.generation_number % self.settings.STATUS_PERIOD == 0:
			print(self.get_progress_status())

			if self.evolution.get_population_best()['fitness'] != self.last_max_fitness:
				self.last_max_fitness = self.evolution.get_population_best()['fitness']
				self.img.set_title(self.get_picture_title())

				if self.settings.num_of_layers == 1:
					self.img.imshow(self.evolution.get_population_best()['pic'][0], cmap='gray', interpolation='none')
				else:
					self.img.imshow(np.swapaxes(self.evolution.get_population_best()['pic'], 0, 2), interpolation='none')

				plt.savefig(self.settings.RESULTS_PATH)

			if self.evolution.termination_condition():
				self.animation.event_source.stop()

		self.generation_number += 1

	def init(self):
		self.line.set_data([], [])
		return self.line,

	def animate_graph(self, i):
		self.evolution.create_new_generation()

		self.graph_x.append(i)
		self.graph_y.append(self.evolution.get_population_best()['fitness'])
		self.reset_plot_bounds()

		if self.generation_number % self.settings.STATUS_PERIOD == 0:
			print(self.get_progress_status())

			self.line.set_data(self.graph_x, self.graph_y)
			self.graph.figure.canvas.draw()

			if self.evolution.get_population_best()['fitness'] != self.last_max_fitness:
				self.last_max_fitness = self.evolution.get_population_best()['fitness']
				self.img.set_title(self.get_picture_title())
				if self.settings.num_of_layers == 1:
					self.img.imshow(self.evolution.get_population_best()['pic'][0], cmap='gray', interpolation='none')
				else:
					self.img.imshow(self.evolution.get_population_best()['pic'], interpolation='none')

				plt.savefig(self.settings.RESULTS_PATH)

			if self.evolution.termination_condition():
				self.animation.event_source.stop()

		self.generation_number += 1

		return self.line,

	def reset_plot_bounds(self):
		x_max = self.graph.get_xlim()[1]
		if len(self.graph_x) >= x_max:
			self.graph.set_xlim(0, 2 * x_max)

		y_max = self.graph.get_ylim()[1]
		if self.graph_y[len(self.graph_y) - 1] >= y_max:
			self.graph.set_ylim(0, 2 * y_max)

	def get_picture_title(self):
		title = f"GEN:{self.generation_number}, Label:{self.settings.label}\n"

		if self.settings.penalty_factor != 0:
			return title + "Certainty:{0:.2f}%".format(round(self.evolution.get_population_best()['certainty']*100, 2)) +\
				   ", Penalty:{0:.4f}".format(round(self.evolution.get_population_best()['penalty'], 4)) + \
				   ", Fitness:{0:.4f}".format(round(self.evolution.get_population_best()['fitness'], 4))
		else:
			return title + "Fitness:{0:.2f}%".format(round(self.evolution.get_population_best()['fitness']*100, 2))

	def get_progress_status(self):
		status = f"GEN:{self.generation_number}, Label:{self.settings.label} | BEST("

		if self.settings.penalty_factor != 0:
			status += "Certainty:{0:.4f}".format(round(self.evolution.get_population_best()['certainty'], 4)) + \
				   ", Penalty:{0:.4f}".format(round(self.evolution.get_population_best()['penalty'], 4)) + \
				   ", Fitness:{0:.4f}".format(round(self.evolution.get_population_best()['fitness'], 4))
		else:
			status += "Fitness:{0:.4f}".format(round(self.evolution.get_population_best()['fitness'], 4))

		status += ") | Worst(Fitness:{0:.4f})".format(round(self.evolution.get_population_wrost()['fitness'], 4))
		return status

class GreedyAlgorithm:
	def greedy_optimization(self):
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

		plt.imshow(pic[0], cmap='gray', interpolation='none')
		plt.axis("off")
		plt.savefig(self.settings.RESULTS_PATH)

	def certainty(self, pic: np.array):
		with torch.no_grad():
			network_input = torch.Tensor(pic).unsqueeze(dim=0)  # Expending the array (one example)
			network_output = self.settings.network(network_input)
			certainty = torch.nn.functional.softmax(network_output, dim=1)

			return certainty[0][self.settings.label_num]