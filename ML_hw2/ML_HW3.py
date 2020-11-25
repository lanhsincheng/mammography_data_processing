import numpy as np
import matplotlib.pyplot as plt


def mixed_gaussian(sigmas, means, priors, n = 10000):

    data_list = []
    for sigma, mean, prior in zip(sigmas, means, priors):
        m = np.random.normal(mean, sigma, n)
        data_list.append(m)
        count, bins, ignored = plt.hist(m, 100, density=True)
        plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mean) ** 2 / (2 * sigma ** 2)), linewidth=2, color='r')
    # plt.show()

    mm = np.concatenate((data_list[0], data_list[1]), axis=None)
    # for sigma, mean, prior in zip(sigmas, means, priors):
    #     count, bins, ignored = plt.hist(mm, 100, density=True)
    #     plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mean) ** 2 / (2 * sigma ** 2)), linewidth=2,
    #              color='r')
    # plt.show()

    return mm, means

def distance_metric(data, randompoint):
    return 0.5 * (data - randompoint) ** 2
    # return (data - randompoint) ** 2

def hardKmeans(distance_table):
    # separate into 0,1
    # check wether [0] or [1] is cloder to 2 random points
    return distance_table == np.min(distance_table, axis = 1, keepdims= True)

def softKmeans(distance_table, beta = 2):
    # separate into 0-1
    distance_table = np.exp(-beta * distance_table) / np.exp(-beta * distance_table).sum(axis = 1)[:,None]
    return distance_table

def kmeans(mm, k, func, termination = 0.001):

    randompoint_list = np.random.choice(mm, k, replace=False)
    # k-means update process
    while True:
        distance_table = np.array([[distance_metric(data, randompoint) for randompoint in randompoint_list] for data in mm])
        # return new cluster classes
        new_mm_cluster = func(distance_table)
        # k = mm[:, None] * new_mm_cluster
        new_randompoint_list = (mm[:, None] * new_mm_cluster).sum(axis=0) / new_mm_cluster.sum(axis=0)

        if np.abs((new_randompoint_list - randompoint_list) / randompoint_list).sum() < termination:
            return randompoint_list

        randompoint_list = new_randompoint_list

def demo(params):
    # fig = plt.figure()
    mm, means = mixed_gaussian(**params)
    k = 2
    x1, x2 = kmeans(mm, k, hardKmeans)
    plt.axvline(x=x1, ymax = 0.9, color="k", label=f"hard: {round(x1,3)}, {round(x2,3)}")
    plt.axvline(x=x2, ymax = 0.9, color="k")

    y1, y2 = kmeans(mm, k, softKmeans)
    plt.axvline(x=y1, ymax = 0.9, color="c", linewidth = 2, linestyle ="dotted", label=f"soft: {round(y1,3)}, {round(y2,3)}")
    plt.axvline(x=y2, ymax = 0.9, color="c", linewidth = 2, linestyle ="dotted",)

    plt.axvline(x=means[0], ymax = 0.5, linewidth = 3, linestyle ="--", color="g", label=f"true mean: {means[0]}, {means[1]}")
    plt.axvline(x=means[1], ymax = 0.5, linewidth = 3, linestyle ="--", color="g")

    plt.legend(loc='upper left', fontsize='xx-small', frameon=False)


# 2 gaussian model params
params_li = [ {"sigmas": [1,1], "means": [-0.5, 0.5], "priors": [0.5, 0.5]},
              {"sigmas": [1,1], "means": [-1, 1], "priors": [0.5, 0.5]},
              {"sigmas": [1,1], "means": [-2, 2], "priors": [0.5, 0.5]},
              ]

plt.gcf().set_size_inches(5*len(params_li), 4)
# fig = plt.figure()
for i, params in enumerate(params_li):
    plt.subplot(1, len(params_li), i+1)
    demo(params)
plt.show()