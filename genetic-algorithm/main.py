from matplotlib import pyplot as plt
from matplotlib import animation

from evolution import Evolution


class Main:
    x = []
    y = []

    def __init__(self):
        self.evolution = Evolution()
        fig = plt.figure()
        ax = plt.axes(xlim=(0, 10000), ylim=(0, 10))
        self.line, = ax.plot([], [], lw=2)
        animation.FuncAnimation(fig, self.animate, init_func=self.init,
                                frames=10000, interval=0, blit=True)
        plt.show()

    def init(self):
        self.line.set_data([], [])
        return self.line,

    def animate(self, i):
        self.x.append(i)
        self.y.append(self.evolution.create_new_generation())
        self.line.set_data(self.x, self.y)
        return self.line,


Main()
