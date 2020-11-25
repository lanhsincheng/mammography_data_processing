import scipy
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import operator
import functools
# def factorial(n: int) -> int:
#     return n * factorial(n - 1) if n > 1 else 1
def factorial(n):
    return functools.reduce(operator.mul, range(1, n+1))


def combination(n: int, k: int) -> int:
    return factorial(n) // factorial(k) // factorial(n - k)


def threshold(n: int, p: float, e: float) -> int:
    return np.ceil(n * p * (1 - e)).astype(int), np.floor(n * p * (1 + e)).astype(int)


def binomial_pmf(n: int, k: int, p: float) -> float:
    return combination(n, k) * p ** k * (1 - p) ** (n - k)


def binomial_cdf(n: int, p: float, e: float) -> float:
    l, u = threshold(n, p, e)
    return sum(binomial_pmf(n, i, p) for i in range(l, u + 1))


def normal_cdf(n: int, p: float, e: float) -> float:
    '''
    gaussian approx. of binomial for prevention of max. recursion depth exceeded
    0.5 accounts for the continuity correction
    '''
    l, u = threshold(n, p, e)
    mu = n * p
    sigma = (n * p * (1 - p)) ** 0.5
    z1 = (l - 0.5 - mu) / sigma
    z2 = (u + 0.5 - mu) / sigma
    return st.norm.cdf(z2) - st.norm.cdf(z1)


def binomial_scipy(n: int, p: float, e: float) -> float:
    l, u = threshold(n, p, e)
    return sum(scipy.stats.binom.pmf(range(l, u + 1), n, p))


def grid(series_n, series_pe, f):
    return np.array([[np.round(f(n, p, e), 4) for p, e in series_pe] for n in series_n])


print(grid(
    series_n=[100, 3000, 50000],
    series_pe=[(0.25, 0.01), (0.05, 0.05)],
    f=binomial_scipy
))
k = factorial(20000)
print()