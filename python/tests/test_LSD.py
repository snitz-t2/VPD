import sys
import cv2
import pytest
import numpy as np
from os.path import join, abspath, dirname

my_dir = dirname(abspath(__file__))
data_folder = join(my_dir, 'input_data')
sys.path.append(dirname(my_dir))

from LSD import LSDWrapper


class TestLSDWrapper(object):
    """
    Tests for the function `LSDWrapper.run_lsd`.
    The function detects line segments in a single image.
    More information can be found in the function itself or in the article.
    Test takes ground-truth information from MATLAB (output) and verifies correctness.
    """
    def test__LSDWrapper_full(self):
        """
        This test verifies that running the LSD algorithm on the test image returns exacly the same results as running
        it in MATLAB.
        """
        # 0) load input data (image) and output data (algorithm results for all lines)
        image_in = cv2.imread(join(data_folder, 'test.jpg'))
        lines_out = np.loadtxt(join(data_folder, 'test__LSDWrapper_full__lines_out.txt'), delimiter=',')

        # 1) call the wrapper
        lsd = LSDWrapper()

        # 2) run the algorithm
        lines = lsd.run_lsd(image_in, cut_results=False)

        # 3) assert results are correct
        assert np.all(np.abs(lines - lines_out) < 1e-12)
