from matplotlib import pyplot as plt
from matplotlib import animation

from evolution import Evolution
from population import Population


class Main:
    x = []
    y = []

    def __init__(self):
        self.evolution = Evolution()

        fig = plt.figure()
        fig.suptitle('genetic algorithm', fontsize=12, fontweight='bold')

        self.ax = fig.add_subplot(111)
        self.ax.axis([0, 10, 0, 10])
        fig.subplots_adjust(top=0.85)
        self.ax.set_title(self.get_title())
        self.line, = self.ax.plot([], [], lw=2)
        self.animation = animation.FuncAnimation(fig, self.animate, init_func=self.init, interval=0, blit=True)
        plt.show()

    def init(self):
        self.line.set_data([], [])
        return self.line,

    def animate(self, i):
        self.evolution.create_new_generation()

        self.x.append(i)
        self.y.append(self.evolution.get_population_max_fitness()['fitness_value'])

        self.reset_plot_bounds()
        self.line.set_data(self.x, self.y)
        self.ax.set_title(self.get_title())
        self.ax.figure.canvas.draw()

        if self.evolution.termination_condition():
            print(self.evolution.get_population_max_fitness())
            self.animation.event_source.stop()

        return self.line,

    def reset_plot_bounds(self):
        x_max = self.ax.get_xlim()[1]
        if len(self.x) >= x_max:
            self.ax.set_xlim(0, 2 * x_max)

        y_max = self.ax.get_ylim()[1]
        if self.y[len(self.y) - 1] >= y_max:
            self.ax.set_ylim(0, 2 * y_max)

    def get_title(self):
        return ('x: ' + str(self.evolution.get_population_max_fitness()[Population.get_population_columns_names()].mean()) +
                '   y: ' + str(self.evolution.get_population_max_fitness()['fitness_value']))


Main()
