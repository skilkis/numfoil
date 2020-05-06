import numpy as np
import pytest

from gammapy.geometry import Airfoil, NACA4Airfoil


class TestAirfoil:

    test_class = Airfoil

    def test_abc(self):
        """Tests if the ABC module works as expected."""

        # Getting abstract methods from current `test_class`
        abstractmethods = ["camber_at", "surfaces_at"]

        # Checking if TypeError is raised for all abstractmethods
        with pytest.raises(TypeError) as e:
            self.test_class()
            assert all(m in str(e.value) for m in abstractmethods)

    SAMPLE_POINTS_CASES = {
        "argnames": "func_args, expected_result",
        "argvalues": [
            # Checking if n_points = 0 works for both "cosine" case
            # Also checks if spacing is indeed an optional input
            ((0,), np.array([])),
            # Checking if n_points = 0 for "linear" case
            ((0, "linear"), np.array([])),
            # Checking cosine spacing with n_points = 5
            # Also checks if the input "cosine" functions as expected
            ((5, "cosine"), np.array([0, 0.14644, 0.5, 0.85355, 1])),
            # Checking if linear spacing works with n_points = 5
            ((5, "linear"), np.array([0, 0.25, 0.5, 0.75, 1])),
            # Checking if ValueError is raised with invalid spacing
            pytest.param(
                (0, None), None, marks=pytest.mark.xfail(raises=ValueError)
            ),
        ],
    }

    @pytest.mark.parametrize(**SAMPLE_POINTS_CASES)
    def test_sample_points(self, func_args, expected_result):
        """Tests if chord-point sampling functions correctly."""
        result = self.test_class.get_sample_points(*func_args)
        assert np.allclose(result, expected_result, atol=1e-5)

    LRU_CACHE_CASES = {
        "argnames": "first_call, second_call, miss",
        "argvalues": [
            # Checking if optional argument matters
            ((0,), (0, "cosine"), False),
            # Checking if cache correctly detects difference of spacing
            ((5, "cosine"), (5, "linear"), False),
        ],
    }

    @pytest.mark.parametrize(**LRU_CACHE_CASES)
    def test_sample_points_lru_cache(self, first_call, second_call, miss):
        """Tests if sampling results are cached correctly."""
        sample_points = self.test_class.get_sample_points

        # Clearning the cache to reset statistics
        sample_points.cache_clear()

        # Performing first-call and checking if cache is updated
        first_result = sample_points(*first_call)
        assert sample_points.cache_info().currsize == 1

        # Performing second-call, checking if cache was correctly
        # used (hit or miss), and checking resultant values
        second_result = sample_points(*first_call)
        if miss:
            assert sample_points.cache_info().currsize == 2
            assert sample_points.cache_info().misses == 1
            assert not np.allclose(first_result, second_result)
        else:
            assert sample_points.cache_info().currsize == 1
            assert sample_points.cache_info().hits == 1
            assert np.allclose(first_result, second_result)


class AirfoilTester:

    test_class = None

    def test_abc(self):
        """Tests if the all abstract methods have been implemented."""
        self.test_class()

    # @pytest.fixture(params=)
    # def airfoils(self, args, kwargs):
    #     return self.test_class(*args, **kwargs)


class TestNACA4Airfoil(AirfoilTester):

    test_class = NACA4Airfoil
    test_scenarios = {
        ""
    }

    # @pytest.mark.parametrize(**)
    # def test_input(self, args, kwargs):


# TO_ZENITH_TEST_CASES = {
#     "argnames": "altitude, expected_result",
#     "argvalues": [
#         (0, 90),
#         (90, 0),
#         (75, 15),
#         pytest.param(
#             180, None, marks=pytest.mark.xfail(raises=ValueError)
#         ),
#     ],
# }

# @pytest.mark.parametrize(**TO_ZENITH_TEST_CASES)
# def test_to_zenith(self, altitude, expected_result):
#     """Tests the altitude to zenith conversion with edge-cases."""
#     assert self.test_class.to_zenith(altitude) == expected_result
