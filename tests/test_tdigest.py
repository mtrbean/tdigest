import pytest
import numpy as np
from scipy.stats.distributions import gamma, uniform

from tdigest.tdigest import TDigest, Centroid


@pytest.fixture()
def empty_tdigest():
    return TDigest()


@pytest.fixture()
def example_positive_centroids():
    mean = np.array([0.5, 1.1, 1.5])
    weight = np.ones_like(mean)
    return mean, weight


@pytest.fixture()
def example_centroids():
    mean = np.array([-1.1, -0.5, 0.1, 1.5])
    weight = np.ones_like(mean)
    return mean, weight


@pytest.fixture()
def example_random_data():
    return np.random.randn(100)


class TestTDigest():

    def test_compute_centroid_quantile(self, empty_tdigest, example_centroids):
        empty_tdigest._means, empty_tdigest._weights = example_centroids
        empty_tdigest.n = 4
        empty_tdigest.m = 4

        assert empty_tdigest._centroid_quantile(0) == (1 / 2. + 0) / 4
        assert empty_tdigest._centroid_quantile(1) == (1 / 2. + 1) / 4
        assert empty_tdigest._centroid_quantile(2) == (1 / 2. + 2) / 4
        assert empty_tdigest._centroid_quantile(3) == (1 / 2. + 3) / 4

    @pytest.mark.xfail
    def test_compress(self, empty_tdigest, example_random_data):
        empty_tdigest.batch_update(example_random_data)
        precompress_n, precompress_len = empty_tdigest.n, len(empty_tdigest)
        empty_tdigest.compress()
        postcompress_n, postcompress_len = empty_tdigest.n, len(empty_tdigest)
        assert postcompress_n == precompress_n
        assert postcompress_len <= precompress_len

    def test_data_comes_in_sorted_does_not_blow_up(self, empty_tdigest):
        t = TDigest()
        for x in range(10000):
            t.update(x, 1)

        assert t.m < 5000

        t = TDigest()
        t.batch_update(range(10000))
        assert t.m < 1000

    def test_extreme_percentiles_return_min_and_max(self, empty_tdigest):
        t = TDigest()
        data = np.random.randn(100000)
        t.batch_update(data)
        assert t.percentile(0) == data.min()
        assert t.percentile(100.) == data.max()

    @pytest.mark.xfail
    def test_update(self):
        c = Centroid(0, 0)
        value, weight = 1, 1
        c.update(value, weight)
        assert c.count == 1
        assert c.mean == 1

        value, weight = 2, 1
        c.update(value, weight)
        assert c.count == 2
        assert c.mean == (2 + 1.) / 2.

        value, weight = 1, 2
        c.update(value, weight)
        assert c.count == 4
        assert c.mean == 1 * 1 / 4. + 2 * 1 / 4. + 1 * 2 / 4.


class TestStatisticalTests():

    def test_uniform(self):
        t = TDigest()
        x = np.random.random(size=10000)
        t.batch_update(x)

        percentile = [50, 10, 90, 1, 99, 0.1, 99.9]
        tolerance = [0.02, 0.01, 0.01, 0.005, 0.005, 0.001, 0.001]
        for q, tol in zip(percentile, tolerance):
            assert abs(t.percentile(q) - q / 100) < tol

    def test_ints(self):
        t = TDigest()
        t.batch_update([1, 2, 3])
        assert t.percentile(50) == 2

        t = TDigest()
        x = [1, 2, 2, 2, 2, 2, 2, 2, 3]
        t.batch_update(x)
        assert t.percentile(50) == 2
        assert t.n == len(x)


