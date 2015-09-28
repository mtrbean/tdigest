from __future__ import division
from copy import deepcopy
import numpy as np


class TDigest(object):
    """ Implementation of Ted Dunning's t-digest data structure.

    The t-digest data structure is designed around computing accurate estimates
    from either streaming data, or distributed data. These estimates are
    percentiles, quantiles, trimmed means, etc. Two t-digests can be added,
    making the data structure ideal for map-reduce settings.
    """

    def __init__(self, delta=0.01, K=25):
        threshold = int(K / delta) + 1
        self.__means = np.empty(threshold)
        self.__weights = np.zeros(threshold)
        self.m = 0
        self.n = 0
        self.delta = delta
        self.K = K

    def __add__(self, other_digest):
        new_digest = deepcopy(self)
        p = np.random.permutation(other_digest.m)
        for i in p:
            new_digest.update(other_digest._means[i], other_digest._weights[i])
        return new_digest

    @property
    def _means(self):
        return self.__means[:self.m]

    @property
    def _weights(self):
        return self.__weights[:self.m]

    def _centroid_quantile(self, i):
        assert 0 <= i < self.m
        d = self._weights[i] / 2 + self._weights[:max(0, i - 1)].sum()
        assert 0 <= d <= self.n
        return d / self.n

    def _sort_centroids(self):
        idx = np.argsort(self._means)
        self._means[:] = self._means[idx]
        self._weights[:] = self._weights[idx]

    def _weight_bound(self, q):
        # return 4 * self.n * self.delta * q * (1 - q)
        return 2 * self.n * self.delta * np.sqrt(q * (1 - q))

    def update(self, x, w=1):
        """
        Update the t-digest with value x and weight w.
        """
        if w == 0:
            return

        self.n += w

        if self.m == 0:
            self.m = 1
            self._means[0] = x
            self._weights[0] = w
            return

        i = np.abs(self._means - x).argmin()
        q = self._centroid_quantile(i)
        bound = self._weight_bound(q)
        if self._weights[i] + w < bound:
            dw = min(w, bound - self._weights[i])
            self._weights[i] += dw
            self._means[i] += dw * (x - self._means[i]) / self._weights[i]
            w -= dw
        if w > 0:
            self.m += 1
            self._means[-1] = x
            self._weights[-1] = w
        if self.m > self.K / self.delta:
            self.compress()

    def batch_update(self, x, w=1):
        """
        Update the t-digest with an iterable of values. This assumes all points
        have the same weight.
        """
        for xx in x:
            self.update(xx, w)

    def compress(self):
        digest = TDigest(self.delta, self.K)
        p = np.random.permutation(self.m)
        for i in p:
            digest.update(self._means[i], self._weights[i])
        self.m = digest.m
        self._means[:] = digest.means
        self._weights[:] = digest.weights

    def percentile(self, q):
        """
        Computes the percentile of a specific value in [0,100].
        """
        if not (0 <= q <= 100):
            raise ValueError("q must be between 0 and 100.")

        self._sort_centroids()
        q *= self.n / 100
        return np.interp(q, self._weights.cumsum(), self._means)

    def cdf(self, x):
        """
        Computes the quantile of a specific value, ie. computes F(q) where F
        denotes the CDF of the distribution.
        """

        self._sort_centroids()
        return np.interp(x, self._means, self._weights.cumsum() / self.n,
                         left=0, right=1)

    def trimmed_mean(self, p1, p2):
        """
        Computes the mean of the distribution between the two percentiles.
        """
        if not (0 <= p1 < p2 <= 100):
            raise ValueError("p1 must be between 0 and 100 and less than p2.")
        self._sort_centroids()
        p1 /= 100
        p2 /= 100
        w = self._weights / self.n
        cum_w = self._weights.cumsum() / self.n
        i1, i2 = np.searchsorted(cum_w, [p1, p2])
        w[i1] = cum_w[i1] - p1
        w[i2] -= cum_w[i2] - p2
        s = slice(i1, i2 + 1)
        return np.dot(w[s], self._means[s]) / w[s].sum()


if __name__ == '__main__':
    from scipy.stats.distributions import gamma, uniform

    dists = [uniform(0, 1), gamma(0.1, scale=10)]

    for dist in dists:
        print(dist.dist.name)
        digest = TDigest()
        digest.batch_update(dist.rvs(size=10000))

        for q in [50, 10, 90, 1, 0.1, 0.01]:
            print(q, dist.ppf(q / 100),
                  abs(digest.percentile(q) - dist.ppf(q / 100)))
