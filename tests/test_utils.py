import numpy as np
from scipy.stats import t as t_dist, sem


def normalize_for_embedding(X):
    X_min, X_max = X.min(axis=0), X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1.0
    return 2 * np.pi * (X - X_min) / X_range


def ci_95(data):
    mean = np.mean(data)
    sem_val = sem(data)
    ci = sem_val * t_dist.ppf((1 + 0.95) / 2., len(data) - 1)
    return mean, mean - ci, mean + ci


def test_normalize_range():
    X = np.array([[0.0, 10.0], [2.0, 20.0], [4.0, 30.0]])
    X_norm = normalize_for_embedding(X)
    assert X_norm.min() >= 0.0, f"Min {X_norm.min()} < 0"
    assert X_norm.max() <= 2 * np.pi + 1e-6, f"Max {X_norm.max()} > 2π"


def test_normalize_output_values():
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    X_norm = normalize_for_embedding(X)
    np.testing.assert_array_almost_equal(X_norm[0], [0.0, 0.0])
    np.testing.assert_array_almost_equal(X_norm[1], [2 * np.pi, 2 * np.pi])


def test_normalize_single_value():
    X = np.array([[5.0], [5.0], [5.0]])
    X_norm = normalize_for_embedding(X)
    np.testing.assert_array_almost_equal(X_norm, [[0.0], [0.0], [0.0]])


def test_normalize_identity():
    X = np.array([[0.0], [2 * np.pi]])
    X_norm = normalize_for_embedding(X)
    np.testing.assert_array_almost_equal(X_norm, [[0.0], [2 * np.pi]])


def test_normalize_preserves_order():
    rng = np.random.default_rng(42)
    X = rng.uniform(-10, 10, (20, 3))
    X_norm = normalize_for_embedding(X)
    for col in range(X.shape[1]):
        order = np.argsort(X[:, col])
        diffs = np.diff(X_norm[order, col])
        assert np.all(diffs >= -1e-12), f"Columna {col}: orden no preservado"


def test_ci_95_symmetry():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mean, low, high = ci_95(data)
    assert abs(mean - (low + high) / 2) < 1e-10


def test_ci_95_constant():
    data = np.ones(10) * 5.0
    mean, low, high = ci_95(data)
    assert mean == 5.0
    assert abs(low - 5.0) < 1e-10
    assert abs(high - 5.0) < 1e-10


def test_ci_95_known():
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, 100)
    mean, low, high = ci_95(data)
    assert low <= mean <= high


def test_ci_95_increases_with_std():
    data1 = np.array([1.0, 2.0, 3.0])
    data2 = np.array([0.0, 2.0, 4.0])
    _, _, high1 = ci_95(data1)
    _, _, high2 = ci_95(data2)
    assert high2 - np.mean(data2) > high1 - np.mean(data1)
