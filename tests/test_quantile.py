# -*- coding: utf-8 -*-

import numpy as np
import pytest

import corner


def test_invalid_quantiles_1(seed=42):
    np.random.seed(seed)
    with pytest.raises(ValueError):
        corner.quantile(np.random.rand(100), [-0.1, 5])


def test_invalid_quantiles_2(seed=42):
    np.random.seed(seed)
    with pytest.raises(ValueError):
        corner.quantile(np.random.rand(100), 5)


def test_invalid_quantiles_3(seed=42):
    np.random.seed(seed)
    with pytest.raises(ValueError):
        corner.quantile(np.random.rand(100), [0.5, 1.0, 8.1])


def test_dimension_mismatch(seed=42):
    np.random.seed(seed)
    with pytest.raises(ValueError):
        corner.quantile(
            np.random.rand(100), [0.1, 0.5], weights=np.random.rand(3)
        )


def test_valid_quantile(seed=42):
    np.random.seed(seed)
    x = np.random.rand(25)
    q = np.arange(0.1, 1.0, 0.111234)

    a = corner.quantile(x, q)
    b = np.percentile(x, 100 * q)
    assert np.allclose(a, b)


def test_weighted_quantile(seed=42):
    np.random.seed(seed)
    x = np.random.rand(25)
    q = np.arange(0.1, 1.0, 0.111234)
    a = corner.quantile(x, q, weights=np.ones_like(x))
    b = np.percentile(x, 100 * np.array(q))
    assert np.allclose(a, b)

    q = [0.0, 1.0]
    a = corner.quantile(x, q, weights=np.random.rand(len(x)))
    assert np.allclose(a, (np.min(x), np.max(x)))
