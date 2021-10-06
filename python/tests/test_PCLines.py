import sys
import pytest
import numpy as np
from os.path import join, abspath, dirname

my_dir = dirname(abspath(__file__))
data_folder = join(my_dir, 'input_data')
sys.path.append(dirname(my_dir))

from PCLines import *


class TestPCLines(object):
    """
    Tests for the function `PCLines.PCLines_straight_all` in all four configurations.
    The function converts lines and points into two possible dual spaces.
    More information can be found in the function itself or in the article.
    All tests take ground-truth information from MATLAB (input and output) and verify correctness.
    """
    def test__PCLines_straight_all__straight_lines(self):
        """
        This test verifies that conversion of lines to the `straight` dual space is done correctly.
        """
        # 0) load input and output data
        xy_in = np.loadtxt(join(data_folder, 'test__PCLines_straight_all__straight__lines_in.txt'), delimiter=',')
        u_out = np.loadtxt(join(data_folder, 'test__PCLines_straight_all__straight__lines_u_out.txt'), delimiter=',')
        v_out = np.loadtxt(join(data_folder, 'test__PCLines_straight_all__straight__lines_v_out.txt'), delimiter=',')

        # 1) run the function
        u, v = PCLines_straight_all(xy_in, 'straight')

        # 2) some results returned as `nan` in MATLAB. verify that numpy returns the same results.
        assert np.all(np.isnan(u) == np.isnan(u_out))
        assert np.all(np.isnan(v) == np.isnan(v_out))

        # 3) assert results are correct (where they are not nan)
        u_not_nan = np.bitwise_not(np.isnan(u))
        v_not_nan = np.bitwise_not(np.isnan(v))
        assert np.all(np.abs(u[u_not_nan] - u_out[u_not_nan]) < 1e-10)
        assert np.all(np.abs(v[v_not_nan] - v_out[v_not_nan]) < 1e-10)

    def test__PCLines_straight_all__twisted_lines(self):
        """
        This test verifies that conversion of lines to the `twisted` dual space is done correctly.
        """
        # 0) load input and output data
        xy_in = np.loadtxt(join(data_folder, 'test__PCLines_straight_all__twisted__lines_in.txt'), delimiter=',')
        u_out = np.loadtxt(join(data_folder, 'test__PCLines_straight_all__twisted__lines_u_out.txt'), delimiter=',')
        v_out = np.loadtxt(join(data_folder, 'test__PCLines_straight_all__twisted__lines_v_out.txt'), delimiter=',')

        # 1) run the function
        u, v = PCLines_straight_all(xy_in, 'twisted')

        # 2) some results returned as `nan` in MATLAB. verify that numpy returns the same results.
        assert np.all(np.isnan(u) == np.isnan(u_out))
        assert np.all(np.isnan(v) == np.isnan(v_out))

        # 3) assert results are correct (where they are not nan)
        u_not_nan = np.bitwise_not(np.isnan(u))
        v_not_nan = np.bitwise_not(np.isnan(v))
        assert np.all(np.abs(u[u_not_nan] - u_out[u_not_nan]) < 1e-10)
        assert np.all(np.abs(v[v_not_nan] - v_out[v_not_nan]) < 1e-10)

    def test__PCLines_straight_all__straight_points(self):
        """
        This test verifies that conversion of points to the `straight` dual space is done correctly.
        """
        # 0) load input and output data
        xy_in = np.loadtxt(join(data_folder, 'test__PCLines_straight_all__straight__points_in.txt'), delimiter=',')
        u_out = np.loadtxt(join(data_folder, 'test__PCLines_straight_all__straight__points_u_out.txt'), delimiter=',')
        v_out = np.loadtxt(join(data_folder, 'test__PCLines_straight_all__straight__points_v_out.txt'), delimiter=',')

        # 1) run the function
        u, v = PCLines_straight_all(xy_in, 'straight')

        # 2) some results returned as `nan` in MATLAB. verify that numpy returns the same results.
        assert np.all(np.isnan(u) == np.isnan(u_out))
        assert np.all(np.isnan(v) == np.isnan(v_out))

        # 3) assert results are correct (where they are not nan)
        u_not_nan = np.bitwise_not(np.isnan(u))
        v_not_nan = np.bitwise_not(np.isnan(v))
        assert np.all(np.abs(u[u_not_nan] - u_out[u_not_nan]) < 1e-10)
        assert np.all(np.abs(v[v_not_nan] - v_out[v_not_nan]) < 1e-10)

    def test__PCLines_straight_all__twisted_points(self):
        """
        This test verifies that conversion of points to the `twisted` dual space is done correctly.
        """
        # 0) load input and output data
        xy_in = np.loadtxt(join(data_folder, 'test__PCLines_straight_all__twisted__points_in.txt'), delimiter=',')
        u_out = np.loadtxt(join(data_folder, 'test__PCLines_straight_all__twisted__points_u_out.txt'), delimiter=',')
        v_out = np.loadtxt(join(data_folder, 'test__PCLines_straight_all__twisted__points_v_out.txt'), delimiter=',')

        # 1) run the function
        u, v = PCLines_straight_all(xy_in, 'twisted')

        # 2) some results returned as `nan` in MATLAB. verify that numpy returns the same results.
        assert np.all(np.isnan(u) == np.isnan(u_out))
        assert np.all(np.isnan(v) == np.isnan(v_out))

        # 3) assert results are correct (where they are not nan)
        u_not_nan = np.bitwise_not(np.isnan(u))
        v_not_nan = np.bitwise_not(np.isnan(v))
        assert np.all(np.abs(u[u_not_nan] - u_out[u_not_nan]) < 1e-10)
        assert np.all(np.abs(v[v_not_nan] - v_out[v_not_nan]) < 1e-10)
