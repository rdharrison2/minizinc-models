"""Code to generate all rotations of soma cube pieces as sets of grid points.
"""

import numpy as np
#from pads.Automata import RegExp, LookupNFA

# https://johnrausch.com/PuzzlingWorld/chap03.htm
soma = dict(
    V=np.array([[0,0,0],[1,0,0],[0,0,1]]),
    L=np.array([[0,0,0],[1,0,0],[0,0,1],[0,0,2]]),
    T=np.array([[0,0,0],[0,0,1],[1,0,1],[0,0,2]]),
    Z=np.array([[0,0,0],[0,0,1],[1,0,1],[1,0,2]]),
    A=np.array([[0,0,0],[0,0,1],[0,1,0],[1,1,0]]),
    B=np.array([[0,0,0],[0,1,0],[1,1,0],[1,1,1]]),
    P=np.array([[0,0,0],[0,1,0],[1,1,0],[0,1,1]]),
)

mikusinski = dict(
    A=np.array([[0,0,0],[1,0,0],[0,1,0],[0,2,0]]),
    B=np.array([[0,0,0],[1,0,0],[0,1,0],[0,2,0],[1,0,1]]),
    C=np.array([[0,0,0],[1,0,0],[0,1,0],[1,0,1]]),
    D=np.array([[0,0,0],[1,0,0],[1,1,0],[0,0,1]]),
    E=np.array([[0,0,0],[1,0,0],[2,0,0],[1,1,0],[0,0,1]]),
    F=np.array([[1,0,0],[1,1,0],[0,1,0],[1,0,1],[2,0,1]]),
)


def to_set(points):
    """Converts array of points into frozen set of tuples."""
    return frozenset(to_grid(p) for p in points)


def normalise(points):
    """Translates points to origin."""
    t = np.amin(points.transpose(), axis=1)
    return points - t


def to_grid(points, w=4, d=4):
    """Cartesian x,y,z to 1..W*D*H.

    In [241]: np.dot(L,[1,4,16]) + 1
    Out[241]: array([ 1,  2, 17, 33])
    """
    return np.dot(points,[1,w,w*d]) + 1


def make_all_rotations(points):
    """Returns set of distinct rotations of given points.

    >>> make_all_rotations(A)
    {frozenset({1, 2, 17, 21}),
     frozenset({6, 17, 21, 22}),
     frozenset({1, 2, 5, 18}),
     frozenset({5, 6, 18, 22}),
     frozenset({2, 6, 17, 18}),
     frozenset({2, 5, 6, 21}),
     frozenset({1, 5, 6, 17}),
     frozenset({1, 2, 6, 22}),
     frozenset({5, 17, 18, 21}),
     frozenset({1, 17, 18, 22}),
     frozenset({1, 5, 21, 22}),
     frozenset({2, 18, 21, 22})}

    :return:
    """
    x = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    y = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    z = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    rotations = []
    p2 = None
    for i in range(6):
        if p2 is None:
            p2 = points
        elif i == 4:
            p2 = np.matmul(points, y)
        elif i == 5:
            p2 = np.matmul(points, y.transpose())
        else:
            p2 = np.matmul(p2, x)
        p3 = None
        for j in range(4):
            p3 = p2 if p3 is None else np.matmul(p3, z)
            rotations.append(p3)
    return set(to_set(normalise(r)) for r in rotations)


def to_regex(grid_points):
    """Converts set of ints to regex

    In [272]: to_regex({1, 2, 17, 21})
    Out[272]: '1100000000000000001000001'
    """
    s = []
    m = None
    for n in sorted(grid_points):
        if m and n > m + 1:
            s.append("0{" + str((n - m)-1) + "}")
        s.append("1")
        m = n
    return "0* " + " ".join(s) + " 0*"


def make_all_regex(points):
    rotations = make_all_rotations(points)
    return "|".join([to_regex(p) for p in rotations])


# def regex_to_dfa(regex):
#     """Returns minimized DFA for regex string."""
#     r = RegExp(regex)
#     dfa = renumber_nfa(r.minimize().asNFA())
#     dfa.pprint()
#     return dfa
#
#
# def dfa_to_simple(dfa):
#     alphabet = sorted(dfa.alphabet)
#     num_states = len(dfa.ttable) / len(alphabet)
#     trans = []
#     for s in range(num_states):
#         trans.append([dfa.ttable[s,sym][0] for sym in alphabet])
#     return {'n': num_states, 'alpha': alphabet, 'trans': trans, 'initial': list(dfa.initial)[0], 'final': list(dfa.final)}
#
#
#
# def renumber_nfa(N, offset=1):
#     """Replace NFA state objects with small integers, with 0 for failure state."""
#     replacements = {}  #
#     for x in N.states():
#         # reserve 0 for states that trivially map back to themselves
#         if all({x} == N.transition(x, sym) for sym in N.alphabet):
#             replacements[x] = 0
#         else:
#             replacements[x] = offset
#             offset += 1
#     initial = [replacements[x] for x in N.initial]
#     ttable = {}  # (state, sym) -> list(state)
#     for state in N.states():
#         for symbol in N.alphabet:
#             ttable[replacements[state],symbol] = [replacements[x]
#                 for x in N.transition(state,symbol)]
#     final = [replacements[x] for x in N.states() if N.isfinal(x)]
#     return LookupNFA(N.alphabet,initial,ttable,final)
#
# def points_to_dfa(points):
#     rotations = make_all_rotations(points)
#     regex = "+".join(to_regex(r) for r in rotations)
#     dfa = regex_to_dfa(regex)
#     return (len(dfa.))
