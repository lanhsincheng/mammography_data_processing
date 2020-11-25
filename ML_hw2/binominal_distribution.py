from scipy.stats import binom
import scipy.stats as st
import matplotlib.pyplot as plt
import numpy as np

# n = 100
# p = 0.25
# e = 0.01

def bino_dist(n,p,e):
    mu = n*p
    sigma = np.sqrt(n*(p*(1-p)))

    # binom.cdf
    lcb = int(np.ceil(n*p*(1-e))-0.5)
    ucb = int(np.floor(n*p*(1+e))+0.5)
    # norm.cdf
    lcb_n = np.ceil(n*p*(1-e))
    ucb_n = np.floor(n*p*(1+e))
    z_lcb = (lcb_n-mu-0.5)/sigma
    z_ucb = (ucb_n-mu+0.5)/sigma

    prob = round(binom.cdf(ucb, n, p) - binom.cdf(lcb, n, p), 4)
    prob_n = round(st.norm.cdf(z_ucb) - st.norm.cdf(z_lcb), 4)

    return prob

print('n=100, p=0.25, e=0.01 : ', bino_dist(100,0.25,0.01))
print('n=3000, p=0.25, e=0.01 : ', bino_dist(3000,0.25,0.01))
print('n=50000, p=0.25, e=0.01 : ', bino_dist(50000,0.25,0.01))
print('----------------------------------')
print('n=100, p=0.05, e=0.05 : ', bino_dist(100,0.05,0.05))
print('n=3000, p=0.05, e=0.05 : ', bino_dist(3000,0.05,0.05))
print('n=50000, p=0.05, e=0.05 : ', bino_dist(50000,0.05,0.05))
