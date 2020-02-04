import timeit
import numpy as np
import pandas as pd
from models.enums import Face as F, Columns as C, Action


class State:
    state: np.ndarray
    rightState: np.ndarray

    def __init__(self):
        self.rightState = np.array(
            [
                [F.t, F.t, F.t, F.l, F.l, F.l, F.f, F.f, F.f, F.r, F.r, F.r, F.b, F.b, F.b, F.d, F.d, F.d],
                [F.t, F.t, F.t, F.l, F.l, F.l, F.f, F.f, F.f, F.r, F.r, F.r, F.b, F.b, F.b, F.d, F.d, F.d],
                [F.t, F.t, F.t, F.l, F.l, F.l, F.f, F.f, F.f, F.r, F.r, F.r, F.b, F.b, F.b, F.d, F.d, F.d]
            ]
        )
        self.state = self.rightState

    def action(self, action: Action):
        {
            Action.t: self.t,
            Action.T: self.T,
            Action.l: self.l,
            Action.L: self.L,
            Action.f: self.f,
            Action.F: self.F,
            Action.r: self.r,
            Action.R: self.R,
            Action.b: self.b,
            Action.B: self.B,
            Action.d: self.d,
            Action.D: self.D,
        }.get(action, lambda: None)()

    def t(self):
        self.quartette_swap(((0, C.t0), (0, C.t2), (2, C.t2), (2, C.t0)))
        self.quartette_swap(((0, C.t1), (1, C.t2), (2, C.t1), (1, C.t0)))
        self.quartette_swap(((0, C.f0), (0, C.l0), (0, C.b0), (0, C.r0)))
        self.quartette_swap(((0, C.f1), (0, C.l1), (0, C.b1), (0, C.r1)))
        self.quartette_swap(((0, C.f2), (0, C.l2), (0, C.b2), (0, C.r2)))

    def T(self):
        self.quartette_swap(((2, C.to), (2, C.t2), (0, C.t2), (0, C.t0)))
        self.quartette_swap(((1, C.t0), (2, C.t1), (1, C.t2), (0, C.t1)))
        self.quartette_swap(((0, C.r0), (0, C.b0), (0, C.l0), (0, C.f0)))
        self.quartette_swap(((0, C.r1), (0, C.b1), (0, C.l1), (0, C.f1)))
        self.quartette_swap(((0, C.r2), (0, C.b2), (0, C.l2), (0, C.f2)))

    def l(self):
        self.quartette_swap(((0, C.l0), (0, C.l2), (2, C.l2), (2, C.l0)))
        self.quartette_swap(((0, C.l1), (1, C.l2), (2, C.l1), (1, C.l0)))
        self.quartette_swap(((0, C.b2), (2, C.t0), (2, C.f0), (2, C.d0)))
        self.quartette_swap(((1, C.b2), (1, C.t0), (1, C.f0), (1, C.d0)))
        self.quartette_swap(((2, C.b2), (0, C.t0), (0, C.f0), (0, C.d0)))

    def L(self):
        self.quartette_swap(((2, C.l0), (2, C.l2), (0, C.l2), (0, C.l0)))
        self.quartette_swap(((1, C.l0), (2, C.l1), (1, C.l2), (0, C.l1)))
        self.quartette_swap(((2, C.d0), (2, C.f0), (2, C.t0), (0, C.b2)))
        self.quartette_swap(((1, C.d0), (1, C.f0), (1, C.t0), (1, C.b2)))
        self.quartette_swap(((0, C.d0), (0, C.f0), (0, C.t0), (2, C.b2)))

    def f(self):
        self.quartette_swap(((0, C.f0), (0, C.f2), (2, C.f2), (2, C.f0)))
        self.quartette_swap(((0, C.f1), (1, C.f2), (2, C.f1), (1, C.f0)))
        self.quartette_swap(((2, C.t0), (0, C.r0), (0, C.d2), (2, C.l2)))
        self.quartette_swap(((2, C.t1), (1, C.r0), (0, C.d1), (1, C.l2)))
        self.quartette_swap(((2, C.t2), (2, C.r0), (0, C.d0), (0, C.l2)))

    def F(self):
        self.quartette_swap(((2, C.f0), (2, C.f2), (0, C.f2), (0, C.f0)))
        self.quartette_swap(((1, C.f0), (2, C.f1), (1, C.f2), (0, C.f1)))
        self.quartette_swap(((2, C.l2), (0, C.d2), (0, C.r0), (2, C.t0)))
        self.quartette_swap(((1, C.l2), (1, C.r0), (0, C.d1), (2, C.t1)))
        self.quartette_swap(((0, C.l2), (2, C.r0), (0, C.d0), (2, C.t2)))

    def r(self):
        self.quartette_swap(((0, C.r0), (0, C.r2), (2, C.r2), (2, C.r0)))
        self.quartette_swap(((0, C.r1), (1, C.r2), (2, C.r1), (1, C.r0)))
        self.quartette_swap(((0, C.t2), (2, C.b0), (0, C.d2), (0, C.f2)))
        self.quartette_swap(((1, C.t2), (1, C.b0), (1, C.d2), (1, C.f2)))
        self.quartette_swap(((2, C.t2), (2, C.b0), (2, C.d2), (2, C.f2)))

    def R(self):
        self.quartette_swap(((2, C.r0), (2, C.r2), (0, C.r2), (0, C.r0)))
        self.quartette_swap(((1, C.r0), (2, C.r1), (1, C.r2), (0, C.r1)))
        self.quartette_swap(((0, C.f2), (0, C.d2), (2, C.b0), (0, C.t2)))
        self.quartette_swap(((1, C.f2), (1, C.d2), (1, C.b0), (1, C.t2)))
        self.quartette_swap(((2, C.f2), (2, C.d2), (2, C.b0), (2, C.t2)))

    def b(self):
        self.quartette_swap(((0, C.b0), (0, C.b2), (2, C.b2), (2, C.b0)))
        self.quartette_swap(((0, C.b1), (1, C.b2), (2, C.b1), (1, C.b0)))
        self.quartette_swap(((0, C.t0), (0, C.r2), (2, C.d2), (2, C.l0)))
        self.quartette_swap(((0, C.t1), (1, C.r2), (2, C.d1), (1, C.l0)))
        self.quartette_swap(((0, C.t2), (2, C.r2), (2, C.d0), (0, C.l0)))

    def B(self):
        self.quartette_swap(((2, C.b0), (2, C.b2), (0, C.b2), (0, C.b0)))
        self.quartette_swap(((1, C.b0), (2, C.b1), (1, C.b2), (0, C.b1)))
        self.quartette_swap(((2, C.l0), (2, C.d2), (0, C.r2), (0, C.t0)))
        self.quartette_swap(((1, C.l0), (2, C.d1), (1, C.r2), (0, C.t1)))
        self.quartette_swap(((0, C.l0), (2, C.d0), (2, C.r2), (0, C.t2)))

    def d(self):
        self.quartette_swap(((0, C.d0), (0, C.d2), (2, C.d2), (2, C.d0)))
        self.quartette_swap(((0, C.d1), (1, C.d2), (2, C.d1), (1, C.d0)))
        self.quartette_swap(((2, C.f0), (2, C.r0), (2, C.b0), (2, C.l0)))
        self.quartette_swap(((2, C.f1), (2, C.r1), (2, C.b1), (2, C.l1)))
        self.quartette_swap(((2, C.f2), (2, C.r2), (2, C.b2), (2, C.l2)))

    def D(self):
        self.quartette_swap(((2, C.d0), (2, C.d2), (0, C.d2), (0, C.d0)))
        self.quartette_swap(((1, C.d0), (2, C.d1), (1, C.d2), (0, C.d1)))
        self.quartette_swap(((2, C.l0), (2, C.b0), (2, C.r0), (2, C.f0)))
        self.quartette_swap(((2, C.l1), (2, C.b1), (2, C.r1), (2, C.f1)))
        self.quartette_swap(((2, C.l2), (2, C.b2), (2, C.r2), (2, C.f2)))

    def quartette_swap(self, p):
        temp = self.state[p[3][0], p[3][1].value]
        self.state[p[3][0], p[3][1].value] = self.state[p[2][0], p[2][1].value]
        self.state[p[2][0], p[2][1].value] = self.state[p[1][0], p[1][1].value]
        self.state[p[1][0], p[1][1].value] = self.state[p[0][0], p[0][1].value]
        self.state[p[0][0], p[0][1].value] = temp


state = State()
state.action(Action.f)
