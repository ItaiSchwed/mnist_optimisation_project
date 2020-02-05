from enum import Enum


class Face(Enum):
    t = 0
    l = 1
    f = 2
    r = 3
    b = 4
    d = 5

    def __sub__(self, other):
        if self.name == other.name:
            return 1
        return 0


class Action(Enum):
    f = 1
    F = 2
    b = 3
    B = 4
    r = 5
    R = 6
    l = 7
    L = 8
    t = 9
    T = 10
    d = 11
    D = 12
    freeze = 13


class Columns(Enum):
    t0 = 0
    t1 = 1
    t2 = 2
    l0 = 3
    l1 = 4
    l2 = 5
    f0 = 6
    f1 = 7
    f2 = 8
    r0 = 9
    r1 = 10
    r2 = 11
    b0 = 12
    b1 = 13
    b2 = 14
    d0 = 15
    d1 = 16
    d2 = 17
