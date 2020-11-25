from scipy.stats import binom
from scipy.special import gamma
import scipy.stats as st
import matplotlib.pyplot as plt
import numpy as np
import math

n = 5
p = 0.1

n1 = 50
p1 = 0.1

def pmf_gamma(r, n, p):
    return (gamma(n+1)/(gamma(r+1)*gamma(n-r+1)))*(p**r)*((1-p)**(n-r))

def combination(r, n):
    return (gamma(n + 1) / (gamma(r + 1) * gamma(n - r + 1)))

r_values = list(range(n + 1))
n_r_values = [x/n for x in r_values]
mean, var = binom.stats(n, p)
dist = [pmf_gamma(r, n, p) for r in r_values ]
# print(0.5**10)
log_dist = [math.log(x,10) for x in dist]
prob = [(p**i)*((1-p)**(n-i)) for i in range(n+1)]
log_prob = [math.log(x,10) for x in prob]
combine = [combination(r, n) for r in r_values]
log_combine = [math.log(x,10) for x in combine]

r1_values = list(range(n1 + 1))
n_r1_values = [x/n1 for x in r1_values]
mean, var = binom.stats(n1, p1)
dist1 = [pmf_gamma(r, n1, p1) for r in r1_values ]
log_dist1 = [math.log(x,10) for x in dist1]
prob1 = [(p1**i)*((1-p1)**(n1-i)) for i in range(n1+1)]
log_prob1 = [math.log(x,10) for x in prob1]
combine1 = [combination(r, n1) for r in r1_values]
log_combine1 = [math.log(x,10) for x in combine1]

fig = plt.figure()
plt.gcf().set_size_inches(12,8)
x = np.linspace(0, n, 100)
ax = plt.axes()
plt.ylim(-55, 18)

plt.plot(n_r_values, log_dist, label = f"binomial_n={n}_p={p}")
plt.plot(n_r_values, log_combine, label = f"combine_n={n}_p={p}")
plt.plot(n_r_values, log_prob, label = f"prob_n={n}_p={p}")

plt.plot(n_r1_values, log_dist1, label = f"binomial_n={n1}_p={p1}")
plt.plot(n_r1_values, log_combine1, label = f"combine_n={n1}_p={p1}")
plt.plot(n_r1_values, log_prob1, label = f"prob_n={n1}_p={p1}")

plt.legend()
plt.show()