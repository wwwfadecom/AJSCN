
from sklearn.metrics import pairwise_distances as pair
from sklearn.preprocessing import normalize

import numpy as np


topk = 10

def construct_graph(features, label, method='heat'):
    fname = 'graph/usps10_graph.txt'
    num = len(label)
    dist = None

    if method == 'heat':
        dist = -0.05 * pair(features) ** 2
        dist = np.exp(dist)
    elif method == 'cos':
        features[features > 0] = 1
        dist = np.dot(features, features.T)
    elif method == 'ncos':
        features[features > 0] = 1
        features = normalize(features, axis=1, norm='l1')
        dist = np.dot(features, features.T)

    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk+1))[-(topk+1):]
        inds.append(ind)

    f = open(fname, 'w')
    counter = 0
    A = np.zeros_like(dist)
    for i, v in enumerate(inds):
        mutual_knn = False
        for vv in v:
            if vv == i:
                pass
            else:
                if label[vv] != label[i]:
                    counter += 1
                f.write('{} {}\n'.format(i, vv))
    f.close()
    print('error rate: {}'.format(counter / (num * topk)))

reut = np.loadtxt('data/usps.txt', dtype=float)
label = np.loadtxt('data/usps_label.txt', dtype=int)

construct_graph(reut, label, 'heat')
