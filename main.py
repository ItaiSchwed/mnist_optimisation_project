from matplotlib import pyplot as plt
from matplotlib import animation

import settings
from evolution import Evolution


class Main:
	graph_x = []
	graph_y = []

	def __init__(self):
		self.evolution = Evolution()
		self.evolution.create_new_generation()
		self.last_max_fitness = 0
		self.generation_number = 0

		# Show a picture over time
		if settings.DISPLAY == settings.DISPLAYS.PICTURE_ONLY:
			fig = plt.figure()
			fig.subplots_adjust(top=0.8)
			fig.suptitle('Genetic Algorithm- Transparent Generator', fontsize=12, fontweight='bold')
			self.img = fig.add_subplot(1, 1, 1)
			self.animation = animation.FuncAnimation(fig, self.animate_only_picture, interval=1)
			plt.show()

		# Print in command line (standard output)
		elif settings.DISPLAY == settings.DISPLAYS.COMMAND_LINE:
			while not self.evolution.termination_condition():
				self.evolution.create_new_generation()

				self.generation_number += 1
				if self.generation_number % settings.STATUS_PERIOD == 0:
					print(self.get_progress_status())

					if self.evolution.get_certainty_of_max_fitness() > 0.99:
						plt.imshow(self.evolution.get_population_best_pic(), cmap='gray', interpolation='none')
						plt.show()

		# Show a progress graph
		elif settings.DISPLAY == settings.DISPLAYS.GRAPH:
			fig = plt.figure(figsize=(15, 7))
			fig.suptitle('MNIST- Genetic Algorithm', fontsize=12, fontweight='bold')
			self.graph = fig.add_subplot(1, 2, 1)
			self.graph.set_xlabel("Number of Cross-Overs")
			self.graph.set_ylabel("Fitness")
			self.img = fig.add_subplot(1, 2, 2)

			fig.subplots_adjust(top=0.87)
			self.line, = self.graph.plot([], [], lw=2)

			self.animation = animation.FuncAnimation(fig, self.animate_graph, init_func=self.init, interval=1, blit=True)
			plt.show()

		else:
			print("Illegal Display Option!")
			exit()

	def animate_only_picture(self, i):
		self.evolution.create_new_generation()

		if self.generation_number % settings.STATUS_PERIOD == 0:
			print(self.get_progress_status())

			if self.evolution.get_population_max_fitness() != self.last_max_fitness:
				self.last_max_fitness = self.evolution.get_population_max_fitness()
				self.img.set_title(self.get_picture_title())
				self.img.imshow(self.evolution.get_population_best_pic(), cmap='gray', interpolation='none')

			if self.evolution.termination_condition():
				self.animation.event_source.stop()

		self.generation_number += 1

	def init(self):
		self.line.set_data([], [])
		return self.line,

	def animate_graph(self, i):
		self.evolution.create_new_generation()

		self.graph_x.append(i)
		self.graph_y.append(self.evolution.get_population_max_fitness())
		self.reset_plot_bounds()

		if self.generation_number % settings.STATUS_PERIOD == 0:
			print(self.get_progress_status())

			self.line.set_data(self.graph_x, self.graph_y)
			self.graph.figure.canvas.draw()

			if self.evolution.get_population_max_fitness() != self.last_max_fitness:
				self.last_max_fitness = self.evolution.get_population_max_fitness()
				self.img.set_title(self.get_picture_title())
				self.img.imshow(self.evolution.get_population_best_pic(), cmap='gray', interpolation='none')

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
		title = f"GEN:{self.generation_number}, Label:{settings.LABEL}\n"

		if settings.WITH_PENALTY:
			return title + "Certainty:{0:.2f}%".format(round(self.evolution.get_certainty_of_max_fitness()*100, 2)) +\
				   ", Penalty:{0:.4f}".format(round(self.evolution.get_penalty_of_max_fitness(), 4)) + \
				   ", Fitness:{0:.4f}".format(round(self.evolution.get_population_max_fitness(), 4))
		else:
			return title + "Fitness:{0:.2f}%".format(round(self.evolution.get_population_max_fitness()*100, 2))

	def get_progress_status(self):
		status = f"GEN:{self.generation_number}, Label:{settings.LABEL} | BEST("

		if settings.WITH_PENALTY:
			status += "Certainty:{0:.2f}".format(round(self.evolution.get_certainty_of_max_fitness(), 4)) + \
				   ", Penalty:{0:.4f}".format(round(self.evolution.get_penalty_of_max_fitness(), 4)) + \
				   ", Fitness:{0:.4f}".format(round(self.evolution.get_population_max_fitness(), 4))
		else:
			status += "Fitness:{0:.4f}".format(round(self.evolution.get_population_max_fitness(), 4))

		status += ") | Worst(Fitness:{0:.4f})".format(round(self.evolution.get_population_min_fitness(), 4))
		return status

Main()