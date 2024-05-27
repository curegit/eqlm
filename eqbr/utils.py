import numpy as np

def clamp(a, x, b):
    return min(max(x, a), b)


def lerp(a, b, t):
    return a + t * (b - a)


def ilerp(a, b, x):
    return (x - a) / (b - a)


def weighted_median(values, weights, quantiles=0.5):
    i = np.argsort(values)
    c = np.cumsum(weights[i])
    return values[i[np.searchsorted(c, np.array(quantiles) * c[-1])]]


def chunks(length, count):
    div, mod = divmod(length, count)
    start = 0
    for i in range(count):
        stop = start + div + (1 if i < mod else 0)
        yield start, stop
        start = stop
