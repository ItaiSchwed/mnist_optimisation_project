import timeit

setup = '''
from models.state import State
import random
from models.enums import Columns as C

state = State()
'''

print(timeit.timeit(setup=setup,
                    stmt="state.action(random.choice([c for c in C]))",
                    number=1000) / 1000)
