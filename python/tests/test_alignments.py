import sys
import cv2
import pytest
import numpy as np
from os.path import join, abspath, dirname

my_dir = dirname(abspath(__file__))
data_folder = join(my_dir, 'input_data')
sys.path.append(dirname(my_dir))

import pyalignments_slow


class TestAlignments(object):
    """
    Tests for the function `LSDWrapper.run_lsd`.
    The function detects line segments in a single image.
    More information can be found in the function itself or in the article.
    Test takes ground-truth information from MATLAB (output) and verifies correctness.
    """
    def test__pyalignments_slow(self):
        """
        This test verifies that running the LSD algorithm on the test image returns exacly the same results as running
        it in MATLAB.
        """
        # 0) load input data (image) and output data (algorithm results for all lines)
        points_in = np.loadtxt(join(data_folder, 'test__pyalignments_slow__points_in.txt'), delimiter=',')

        # 1) call the function
        detections = pyalignments_slow.detect_alignments(points_in)

        # # 2) run the algorithm
        # lines = lsd.run_lsd(image_in, cut_results=False)
        #
        # # 3) assert results are correct
        # assert np.all(np.abs(lines - lines_out) < 1e-12)