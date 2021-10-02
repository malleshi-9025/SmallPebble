"""Compute approximate derivatives using finite differences,
useful for testing autodiff gradients.
(And slow to run, hence the need for autodiff.)
"""
import numpy as np


def numgrads(func, args, n=1, delta=1e-6):
    "Numerical nth derivatives of func w.r.t. args."
    gradients = []
    for i, arg in enumerate(args):

        def func_i(a):
            new_args = [x for x in args]
            new_args[i] = a
            return func(*new_args)

        gradfunc = lambda a: numgrad(func_i, a, delta)

        for _ in range(1, n):
            prev_gradfunc = gradfunc
            gradfunc = lambda a: numgrad(prev_gradfunc, a, delta)

        gradients.append(gradfunc(arg))
    return gradients


def numgrad(func, a, delta=1e-6):
    "Numerical gradient of func(a) at `a`."
    grad = np.zeros(a.shape, a.dtype)
    for index, _ in np.ndenumerate(grad):
        delta_array = np.zeros(a.shape, a.dtype)
        delta_array[index] = delta / 2
        grad[index] = np.sum((func(a + delta_array) - func(a - delta_array)) / delta)
    return grad
