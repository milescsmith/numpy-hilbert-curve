import numpy.testing as nptest
import pytest

from hilbert.gray import right_shift


@pytest.mark.parametrize(
    "x,t,k,axs", [
        pytest.param("array_4_ones_1d", "array_7", 1, -1, id="basic_1d"),
        pytest.param("array_4x5_ones_2d", "basic_2d_t_ax1", 1, 1, id="basic_2d_ax1"),
        pytest.param("array_4x5_ones_2d", "basic_2d_t_ax0", 1, 0, id="basic_@d_ax0"),
        pytest.param("array_4_ones_1d", "array_3", 2, -1, id="k2_1d"),
        pytest.param("array_4_ones_1d", "array_4_zeros_1d", 5, -1, id="k2_2d_ax1"),
        pytest.param("array_4x5_ones_2d", "k2_2d_ax1_t", 2, 1, id="k2_2d_ax0"),
        pytest.param("array_4x5_ones_2d", "k2_2d_ax0_t", 2, 0, id="extra_1d"),
        pytest.param("array_4x5_ones_2d", "array_4x5_zeros_2d", 6, 1, id="extra_2d_ax1"),
        pytest.param("array_4x5_ones_2d", "array_4x5_zeros_2d", 6, 0, id="extra_2d_ax0"),
    ]
)
def test_right_shift(x, t, k, axs, request):
    x = request.getfixturevalue(x)
    t = request.getfixturevalue(t)
    y = right_shift(x, k=k, axis=axs)
    nptest.assert_array_almost_equal(y, t)
