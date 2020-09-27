# **<i>Документация</i>**
< br >

### **<i>PAM(k=3, metric="euclidean", max_iter = 300, tol=0.001)</i>**
< br >

### **<i>PARAMETERS</i>**

< ul >
< li > < h3 > < i > k: int, default = 3 < / i > < / h3 > < / li >
The
number
of
clusters
to
form as well as the
number
of
centroids
to
generate.
< li > < h3 > < i > metric: str, default = 'euclidean' < / i > < / h3 > < / li >
Metric
used
to
compute
the
linkage.
< li > < h3 > < i > max_iter: int, default = 300 < / i > < / h3 > < / li >
Maximum
number
of
iterations
of
the
k - means
algorithm
for a single run.
< li > < h3 > < i > tol: float, default = 1e-3 < / i > < / h3 > < / li >
Relative
tolerance
with regards to inertia to declare convergence.
< / ul >

### **<i>ATTRIBUTES</i>**

< ul >
< li > < h3 > < i > inertia_: float < / i > < / h3 > < / li >
Sum
of
squared
distances
of
samples
to
their
closest
cluster
medoid.
< li > < h3 > < i > medoids_: list
of
len(n_clusters) < / i > < / h3 > < / li >
Coordinates
of
cluster
medoids.If
the
algorithm
stops
before
fully
converging(see
tol and max_iter), these
will
not be
consistent
with labels_.
    < li > < h3 > < i > labels_: ndarray
of
shape(n_samples, ) < / i > < / h3 > < / li >
Labels
of
each
point.
< / ul >
< br >
< br >

### **<i>METHOD fit(self, X)</i>**
#### Compute partiton around medoids clustering.
### **<i>PARAMETERS</i>**
< ul >
< li > < h3 > < i > X: array - like or sparse
matrix, shape = (n_samples, n_features) < / i > < / h3 > < / li >
Training
instances
to
cluster.
< / ul >
< br >

### **<i>RETURNS</i>**
< ul >
< li > < h3 > < i > self < / i > < / h3 > < / li >
Fitted
estimator.
< / ul >

## Особенности работы алгоритма описаны комментариями непосредственно в коде
</HTML>


from sklearn.base import BaseEstimator
from itertools import cycle
from math import hypot
import numpy as np
import pandas as pd
import random


def medoid_distribution(med_ind, data):
    dist_vector = np.zeros(len(data))  # ближайшая медоида для каждого объекта
    dist_matr = np.zeros((len(data), len(med_ind)))  # матрица расстояний
    target = np.Inf  # значение целевой функции
    min_dists = np.zeros(len(data))

    for idx, p in enumerate(data):
        ''' поиск для каждого объекта данных ближайшего к нему медоида,
        приписание объекта к этому медоиду: "в матрице расстояний" индексу данного объекта
        присваивается индекс медоида'''
        dist_matr[idx] = [np.linalg.norm(p - data[q]) for q in med_ind]
        dist_vector[idx] = med_ind[np.asarray(dist_matr[idx]).argmin()]
        min_dists[idx] = min(dist_matr[idx])

    target = sum(min_dists)
    return dist_vector, dist_matr, min_dists, target


def medoid_remake(med_ind, dist_vector, data):
    new_meds = []
    for m in med_ind:
        clust = np.where(dist_vector == m)  # индекс тех эл-в в данных, медоид которых равен m
        clust = clust[0]  # функция np.where() возвращает tuple из 1 эл-та, содержащего array
        distances = []  # здесь будут суммы р-й каждой точки до всех остальных в этом кластере
        for i in clust:
            distances.append(sum([np.linalg.norm(data[i] - data[q]) for q in clust if q != i]))
        new_meds.append(clust[np.asarray(distances).argmin()])
    return new_meds


class PAM(BaseEstimator):

    def __init__(self, k=3, metric="euclidean", max_iter=300, tol=0.001):
        self.k = k
        self.metric = metric
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, dt):

        # преобразование данных из dataframe, если необходимо в массив списков характеристик каждого объекта
        if type(dt) == pd.core.frame.DataFrame:
            data = dt[col].values.tolist()
        else:
            data = dt

        # Шаг 1
        med_ind = set()  # рандомная генерация индексов медоидов
        while len(med_ind) != self.k:
            med_ind.add(random.randint(0, len(data) - 1))
        med_ind = list(med_ind)

        # Шаг 2 матрица расстояний
        dist_vector, dist_matr, min_dists, TARGET_INIT = medoid_distribution(med_ind, data)
        # TARGET_INIT - начальное значение целевой ф-и, с которым будем сравнивать её изменения

        # Шаг 3 пересчёт расстояний и медоид
        for i in range(self.max_iter):
            med_ind = medoid_remake(med_ind, dist_vector, data)
            dist_vector, dist_matr, min_dists, target = medoid_distribution(med_ind, data)
            if TARGET_INIT - target < self.tol:
                break

        self.inertia_ = target
        self.medoids_ = med_ind
        self.labels_ = dist_vector

        # return self.inertia_, self.medoids_, self.labels_
