import numpy.random as npr
import numpy.testing as nptest
import pytest

from hilbert.gray import (
    binary2gray,
    gray2binary,
)


@pytest.mark.parametrize(
    "x,t,func", [
        pytest.param("array_4_zeros_1d", "array_4_zeros_1d", binary2gray, id="basic_0000_binary2gray"),
        pytest.param("array_4_zeros_1d", "array_4_zeros_1d", gray2binary, id="basic_0000_gray2binary"),
        pytest.param("array_10","array_4_ones_1d",binary2gray, id="basic_1010_binary2gray"),
        pytest.param("array_4_ones_1d","array_10",gray2binary, id="basic_1010_gray2binary"),
        pytest.param("array_5_ones_1d", "array_16", binary2gray, id="basic_11111_binary2gray"),
        pytest.param("array_16", "array_5_ones_1d", gray2binary, id="basic_11111_gray2binary"),
        pytest.param("array_41","array_61",binary2gray, id="basic_101001_binary2gray"),
        pytest.param("array_61","array_41",gray2binary, id="basic_101001_gray2binary"),
        pytest.param("array_21845","array_32767",binary2gray, id="basic_0101010101010101_binary2gray"),
        pytest.param("array_32767","array_21845",gray2binary, id="basic_0101010101010101_gray2binary"),
    ]
)
def test_conversion(x, t, func, request):
    x = request.getfixturevalue(x)
    t = request.getfixturevalue(t)
    y = func(x)
    nptest.assert_array_equal(y, t)


@pytest.fixture
def mean_val():
    return 30

@pytest.fixture
def bin_len():
    return 0.5

def test_random(mean_val, bin_len):
    npr.seed(1)
    for _ in range(1000):
        length = npr.geometric(1/(mean_val-2)) + 2
        binary1 = npr.rand(length) < bin_len
        gray    = binary2gray(binary1)
        binary2 = gray2binary(gray)
        nptest.assert_equal(binary1, binary2)
