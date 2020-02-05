import time
import timeit
import numpy as np
from models.rubik_state import RubikState
import random
from models.enums import Action

population = np.random.choice([action for action in Action], (1000, 50))

start_time = time.time()
for i in range(1000):
    state = RubikState()
    actions = population[i]
    for action in actions:
        state.action(action)
end_time = time.time()

print((end_time - start_time)/1000)


