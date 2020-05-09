import numpy as np
import pytest

from gammapy.geometry.vector2d import (
    is_row_vector,
    magnitude_2d,
    normalize_2d,
    rotate_2d_90ccw,
)

ROTATE_2D_90CCW_TEST_CASES = {
    "argnames": "array, expected_result, inplace",
    "argvalues": [
        # Testing 1D integer vector
        (np.array([1, 2]), np.array([-2, 1]), False),
        # Testing 2D integer vector
        (np.array([[1, 2]]), np.array([[-2, 1]]), False),
        # Testing that a 2D integer column vector raises ValueError
        pytest.param(
            np.array([[1, 2]]).T,
            None,
            False,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        (
            np.array([2.2, 5.5], dtype=np.float64),
            np.array([-5.5, 2.2], dtype=np.float64),
            False,
        ),
        # Testing matrix of integer row vectors w/ inplace operation
        (
            np.array([[2, 5], [3, 7], [4, 10]]),
            np.array([[-5, 2], [-7, 3], [-10, 4]]),
            True,
        ),
    ],
}


@pytest.mark.parametrize(**ROTATE_2D_90CCW_TEST_CASES)
def test_rotate_90ccw_2D(array, expected_result, inplace):
    """Tests 90 deg rotation affine transform and inplace operation."""
    result = rotate_2d_90ccw(array, inplace=inplace)
    assert np.allclose(result, expected_result)
    assert result is array if inplace else result is not array


MAGNITUDE_2D_TEST_CASES = {
    "argnames": "array, expected_result",
    "argvalues": [
        # Testing that 1D int vector returns 2D float of shape (1, 1)
        (np.array([1, 0]), np.array([[1.0]], dtype=np.float64)),
        # Testing that a 2D integer vector returns a 2D float
        (np.array([[2, 0]]), np.array([[2.0]])),
        # Testing that a 2D integer column vector raises ValueError
        pytest.param(
            np.array([[1, 0]]).T,
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        # Testing an array of 3 row-vectors with negative values
        (
            np.array([[0, -1], [0, -3], [-3, 4]]),
            np.array([[1, 3, 5]], dtype=np.float64).T,
        ),
    ],
}


@pytest.mark.parametrize(**MAGNITUDE_2D_TEST_CASES)
def test_magnitude_2d(array, expected_result):
    result = magnitude_2d(array)
    assert np.allclose(result, expected_result)
    assert result.dtype == expected_result.dtype


NORMALIZE_2D_TEST_CASES = {
    "argnames": "array, expected_result",
    "argvalues": [
        # Testing that 1D int vector returns 2D float of shape (1, 1)
        (np.array([1, 0]), np.array([[1, 0]], dtype=np.float64)),
        # Testing that a 2D integer vector returns a 2D float
        (
            np.array([[-2, 1]]),
            np.array([[-0.8944271912, 0.4472135965]], dtype=np.float64),
        ),
        # Testing that a 2D integer column vector raises ValueError
        pytest.param(
            np.array([[1, 0]]).T,
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        # Testing an array of 3 row-vectors
        (
            np.array([[0, -1], [-3, 0], [-3, 4]]),
            np.array([[0, -1], [-1, 0], [-0.6, 0.8]], dtype=np.float64),
        ),
    ],
}


@pytest.mark.parametrize(**NORMALIZE_2D_TEST_CASES)
def test_normalize_2d(array, expected_result):
    """Tests if 2D row vector(s) are correctly normalized."""
    result = normalize_2d(array)
    assert np.allclose(result, expected_result)
    assert result.dtype == expected_result.dtype


IS_ROW_VECTOR_TEST_CASES = {
    "argnames": "array, expected_result",
    "argvalues": [
        # Testing that 1D vector is treated as a row vector
        (np.array([1, 0]), True),
        # Testing that a 2D column vector raises a ValueError
        pytest.param(
            np.array([[1, 0]]).T,
            True,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        # Testing that 3D row-vector aises an error
        pytest.param(
            np.array([[1, 0, 0]]),
            True,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
    ],
}


@pytest.mark.parametrize(**IS_ROW_VECTOR_TEST_CASES)
def test_is_row_vector(array, expected_result):
    """Tests if column-vectors correctly raise ValueError."""
    result = is_row_vector(array)
    assert result == expected_result
