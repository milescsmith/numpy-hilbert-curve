import numpy as np
import pytest


@pytest.fixture
def array_4_ones_1d():
    return np.ones(4)

@pytest.fixture
def array_5_ones_1d():
    return np.ones(5)

@pytest.fixture
def array_4x5_ones_2d():
    return np.ones((4, 5))


@pytest.fixture
def array_4_zeros_1d():
    return np.zeros(4)


@pytest.fixture
def array_4x5_zeros_2d():
    return np.zeros((4, 5))


@pytest.fixture
def array_7():
    return np.array([0, 1, 1, 1])


@pytest.fixture
def array_3():
    return np.array([0, 0, 1, 1])


@pytest.fixture
def array_10():
    return np.array([1, 0, 1, 0])

@pytest.fixture
def array_16():
    return np.array([1, 0, 0, 0, 0])

@pytest.fixture
def array_41():
    return np.array([1, 0, 1, 0, 0, 1])


@pytest.fixture
def array_61():
    return np.array([1, 1, 1, 1, 0, 1])


@pytest.fixture
def array_21845():
    return np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])


@pytest.fixture
def array_32767():
    return np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])


@pytest.fixture
def basic_2d_t_ax1():
    return np.hstack([np.zeros((4, 1)), np.ones((4, 4))])


@pytest.fixture
def basic_2d_t_ax0():
    return np.vstack([np.zeros((1, 5)), np.ones((3, 5))])


@pytest.fixture
def k2_2d_ax1_t():
    return np.hstack([np.zeros((4, 2)), np.ones((4, 3))])


@pytest.fixture
def k2_2d_ax0_t():
    return np.vstack([np.zeros((2, 5)), np.ones((2, 5))])
