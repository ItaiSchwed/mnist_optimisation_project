from matplotlib import pyplot as plt
from matplotlib import animation

from evolution import Evolution
from models.rubik_state import RubikState
from population import Population


class Main:
    x = []
    y_max = []
    y_min = []
    state: RubikState

    def __init__(self):
        starting_state = RubikState(RubikState.solveState)
        self.evolution = Evolution(starting_state)

        fig = plt.figure()
        fig.suptitle('genetic algorithm', fontsize=12, fontweight='bold')

        self.ax = fig.add_subplot(111)
        self.ax.axis([0, 500, 0, 1])
        fig.subplots_adjust(top=0.85)
        self.ax.set_title(self.get_title())
        self.line_max, = self.ax.plot([], [], lw=2)
        self.line_min, = self.ax.plot([], [], lw=2)
        self.animation = animation.FuncAnimation(fig, self.animate, init_func=self.init, interval=0, blit=True)
        plt.show()

    def init(self):
        self.line_max.set_data(self.x, self.y_max)
        self.line_min.set_data(self.x, self.y_min)
        return self.line_max, self.line_min

    def animate(self, i):
        for j in range(100):
            self.evolution.create_new_generation()

        self.x.append(i * 100)
        self.y_max.append(self.evolution.get_population_max_fitness()['fitness_value'])
        self.y_min.append(self.evolution.get_population_min_fitness()['fitness_value'])

        self.reset_plot_bounds()
        self.line_max.set_data(self.x, self.y_max)
        self.line_min.set_data(self.x, self.y_min)
        self.ax.set_title(self.get_title())
        self.ax.figure.canvas.draw()

        if self.evolution.termination_condition():
            print(self.evolution.get_population_max_fitness())
            self.animation.event_source.stop()

        return self.line_max, self.line_min

    def reset_plot_bounds(self):
        x_max = self.ax.get_xlim()[1]
        if len(self.x)*100 >= x_max:
            self.ax.set_xlim(0, 2 * x_max)

        y_max = self.ax.get_ylim()[1]
        if self.y_max[len(self.y_max) - 1] >= y_max:
            self.ax.set_ylim(0, 2 * y_max)

    def get_title(self):
        # return ('x: ' + str(self.evolution.get_population_max_fitness()[Population.get_population_columns_names()].mean()) +
        #         '   y: ' + str(self.evolution.get_population_max_fitness()['fitness_value']))
        return "dfgdfgdf"


Main()
