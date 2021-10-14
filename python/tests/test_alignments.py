import sys
import cv2
import pytest
import numpy as np
from os.path import join, abspath, dirname

my_dir = dirname(abspath(__file__))
data_folder = join(my_dir, 'input_data')
sys.path.append(dirname(my_dir))

import pyalignments_slow
import pyalignments_fast


class TestAlignments(object):
    """
    Tests for both implementations of the line segments alignment algorithms,
    which are at the core of the vanishing point detection algorithm.

    TODO: add accurate description on functionality
    More information can be found in the function itself or in the article.
    Test takes ground-truth information from MATLAB (output) and verifies correctness.
    """
    def test__pyalignments_slow(self):
        """
        This test verifies that running the slow implementation of the alignments algorithm returns exactly the same
        results as running it in MATLAB.
        """
        # 0) load input data (input points) and output data (line detections)
        points_in = np.loadtxt(join(data_folder, 'test__pyalignments_slow__points_in.txt'), delimiter=',')
        detections_out = np.loadtxt(join(data_folder, 'test__pyalignments_slow__detections_out.txt'), delimiter=',')

        # 1) call the function
        detections = pyalignments_slow.detect_alignments_slow(points_in)

        # 3) assert results are correct
        assert np.all(np.abs(detections_out - detections) < 1e-12)

    def test__pyalignments_fast(self):
        """
        This test verifies that running the fast implementation of the alignments algorithm returns exactly the same
        results as running it in MATLAB.
        """
        # 0) load input data (input points) and output data (line detections)
        points_in = np.loadtxt(join(data_folder, 'test__pyalignments_fast__points_in.txt'), delimiter=',')
        candidate_points_in = np.loadtxt(join(data_folder, 'test__pyalignments_fast__candidate_points_in.txt'),
                                         delimiter=',')
        detections_out = np.loadtxt(join(data_folder, 'test__pyalignments_fast__detections_out.txt'), delimiter=',')

        # 1) call the function
        detections = pyalignments_fast.detect_alignments_fast(points_in, candidate_points_in)

        # 3) assert results are correct
        assert np.all(np.abs(detections_out - detections) < 1e-12)
