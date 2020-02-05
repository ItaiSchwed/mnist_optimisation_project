import numpy as np

from models.enums import Face as F, Columns as C, Action


class RubikState:
    state: np.ndarray
    solveState: np.ndarray = np.array([
                [F.t, F.t, F.t, F.l, F.l, F.l, F.f, F.f, F.f, F.r, F.r, F.r, F.b, F.b, F.b, F.d, F.d, F.d],
                [F.t, F.t, F.t, F.l, F.l, F.l, F.f, F.f, F.f, F.r, F.r, F.r, F.b, F.b, F.b, F.d, F.d, F.d],
                [F.t, F.t, F.t, F.l, F.l, F.l, F.f, F.f, F.f, F.r, F.r, F.r, F.b, F.b, F.b, F.d, F.d, F.d]
            ])

    def __init__(self, starting_state: np.ndarray):
        self.state = np.copy(starting_state)

    def action(self, action: Action):
        {
            Action.t: self._t,
            Action.T: self._T,
            Action.l: self._l,
            Action.L: self._L,
            Action.f: self._f,
            Action.F: self._F,
            Action.r: self._r,
            Action.R: self._R,
            Action.b: self._b,
            Action.B: self._B,
            Action.d: self._d,
            Action.D: self._D,
            Action.freeze: lambda: None
        }.get(action, self._raise_value_error)()

    def _raise_value_error(self):
        raise ValueError("Action doesn't exist")

    def _t(self):
        self._quartette_swap(((0, C.t0), (0, C.t2), (2, C.t2), (2, C.t0)))
        self._quartette_swap(((0, C.t1), (1, C.t2), (2, C.t1), (1, C.t0)))
        self._quartette_swap(((0, C.f0), (0, C.l0), (0, C.b0), (0, C.r0)))
        self._quartette_swap(((0, C.f1), (0, C.l1), (0, C.b1), (0, C.r1)))
        self._quartette_swap(((0, C.f2), (0, C.l2), (0, C.b2), (0, C.r2)))

    def _T(self):
        self._quartette_swap(((2, C.t0), (2, C.t2), (0, C.t2), (0, C.t0)))
        self._quartette_swap(((1, C.t0), (2, C.t1), (1, C.t2), (0, C.t1)))
        self._quartette_swap(((0, C.r0), (0, C.b0), (0, C.l0), (0, C.f0)))
        self._quartette_swap(((0, C.r1), (0, C.b1), (0, C.l1), (0, C.f1)))
        self._quartette_swap(((0, C.r2), (0, C.b2), (0, C.l2), (0, C.f2)))

    def _l(self):
        self._quartette_swap(((0, C.l0), (0, C.l2), (2, C.l2), (2, C.l0)))
        self._quartette_swap(((0, C.l1), (1, C.l2), (2, C.l1), (1, C.l0)))
        self._quartette_swap(((0, C.b2), (2, C.t0), (2, C.f0), (2, C.d0)))
        self._quartette_swap(((1, C.b2), (1, C.t0), (1, C.f0), (1, C.d0)))
        self._quartette_swap(((2, C.b2), (0, C.t0), (0, C.f0), (0, C.d0)))

    def _L(self):
        self._quartette_swap(((2, C.l0), (2, C.l2), (0, C.l2), (0, C.l0)))
        self._quartette_swap(((1, C.l0), (2, C.l1), (1, C.l2), (0, C.l1)))
        self._quartette_swap(((2, C.d0), (2, C.f0), (2, C.t0), (0, C.b2)))
        self._quartette_swap(((1, C.d0), (1, C.f0), (1, C.t0), (1, C.b2)))
        self._quartette_swap(((0, C.d0), (0, C.f0), (0, C.t0), (2, C.b2)))

    def _f(self):
        self._quartette_swap(((0, C.f0), (0, C.f2), (2, C.f2), (2, C.f0)))
        self._quartette_swap(((0, C.f1), (1, C.f2), (2, C.f1), (1, C.f0)))
        self._quartette_swap(((2, C.t0), (0, C.r0), (0, C.d2), (2, C.l2)))
        self._quartette_swap(((2, C.t1), (1, C.r0), (0, C.d1), (1, C.l2)))
        self._quartette_swap(((2, C.t2), (2, C.r0), (0, C.d0), (0, C.l2)))

    def _F(self):
        self._quartette_swap(((2, C.f0), (2, C.f2), (0, C.f2), (0, C.f0)))
        self._quartette_swap(((1, C.f0), (2, C.f1), (1, C.f2), (0, C.f1)))
        self._quartette_swap(((2, C.l2), (0, C.d2), (0, C.r0), (2, C.t0)))
        self._quartette_swap(((1, C.l2), (0, C.d1), (1, C.r0), (2, C.t1)))
        self._quartette_swap(((0, C.l2), (0, C.d0), (2, C.r0), (2, C.t2)))

    def _r(self):
        self._quartette_swap(((0, C.r0), (0, C.r2), (2, C.r2), (2, C.r0)))
        self._quartette_swap(((0, C.r1), (1, C.r2), (2, C.r1), (1, C.r0)))
        self._quartette_swap(((0, C.t2), (2, C.b0), (0, C.d2), (0, C.f2)))
        self._quartette_swap(((1, C.t2), (1, C.b0), (1, C.d2), (1, C.f2)))
        self._quartette_swap(((2, C.t2), (0, C.b0), (2, C.d2), (2, C.f2)))

    def _R(self):
        self._quartette_swap(((2, C.r0), (2, C.r2), (0, C.r2), (0, C.r0)))
        self._quartette_swap(((1, C.r0), (2, C.r1), (1, C.r2), (0, C.r1)))
        self._quartette_swap(((0, C.f2), (0, C.d2), (2, C.b0), (0, C.t2)))
        self._quartette_swap(((1, C.f2), (1, C.d2), (1, C.b0), (1, C.t2)))
        self._quartette_swap(((2, C.f2), (2, C.d2), (0, C.b0), (2, C.t2)))

    def _b(self):
        self._quartette_swap(((0, C.b0), (0, C.b2), (2, C.b2), (2, C.b0)))
        self._quartette_swap(((0, C.b1), (1, C.b2), (2, C.b1), (1, C.b0)))
        self._quartette_swap(((0, C.t0), (0, C.r2), (2, C.d2), (2, C.l0)))
        self._quartette_swap(((0, C.t1), (1, C.r2), (2, C.d1), (1, C.l0)))
        self._quartette_swap(((0, C.t2), (2, C.r2), (2, C.d0), (0, C.l0)))

    def _B(self):
        self._quartette_swap(((2, C.b0), (2, C.b2), (0, C.b2), (0, C.b0)))
        self._quartette_swap(((1, C.b0), (2, C.b1), (1, C.b2), (0, C.b1)))
        self._quartette_swap(((2, C.l0), (2, C.d2), (0, C.r2), (0, C.t0)))
        self._quartette_swap(((1, C.l0), (2, C.d1), (1, C.r2), (0, C.t1)))
        self._quartette_swap(((0, C.l0), (2, C.d0), (2, C.r2), (0, C.t2)))

    def _d(self):
        self._quartette_swap(((0, C.d0), (0, C.d2), (2, C.d2), (2, C.d0)))
        self._quartette_swap(((0, C.d1), (1, C.d2), (2, C.d1), (1, C.d0)))
        self._quartette_swap(((2, C.f0), (2, C.r0), (2, C.b0), (2, C.l0)))
        self._quartette_swap(((2, C.f1), (2, C.r1), (2, C.b1), (2, C.l1)))
        self._quartette_swap(((2, C.f2), (2, C.r2), (2, C.b2), (2, C.l2)))

    def _D(self):
        self._quartette_swap(((2, C.d0), (2, C.d2), (0, C.d2), (0, C.d0)))
        self._quartette_swap(((1, C.d0), (2, C.d1), (1, C.d2), (0, C.d1)))
        self._quartette_swap(((2, C.l0), (2, C.b0), (2, C.r0), (2, C.f0)))
        self._quartette_swap(((2, C.l1), (2, C.b1), (2, C.r1), (2, C.f1)))
        self._quartette_swap(((2, C.l2), (2, C.b2), (2, C.r2), (2, C.f2)))

    def _quartette_swap(self, p):
        temp = self.state[p[3][0], p[3][1].value]
        self.state[p[3][0], p[3][1].value] = self.state[p[2][0], p[2][1].value]
        self.state[p[2][0], p[2][1].value] = self.state[p[1][0], p[1][1].value]
        self.state[p[1][0], p[1][1].value] = self.state[p[0][0], p[0][1].value]
        self.state[p[0][0], p[0][1].value] = temp

    def getSolvingPercent(self):
        diff = 0
        for face in range(6):
            face_column_start = face * 3
            face_column_end = face_column_start + 3
            states = np.array([self.state[:, face_column_start:face_column_end],
                               self.solveState[:, face_column_start:face_column_end]])
            diff += np.sum(np.diff(states, axis=0))
        return diff/54

#
# state = RubikState(RubikState.solveState)
# for action in np.random.choice([action for action in Action], 2):
#     state.action(action)
# print(state.getSolvingPercent())
