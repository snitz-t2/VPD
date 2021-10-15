import sys
import cv2
import pytest
import numpy as np
from os.path import join, abspath, dirname

my_dir = dirname(abspath(__file__))
data_folder = join(my_dir, 'input_data')
sys.path.append(dirname(my_dir))

from denoise_lines import *


class TestDenoiseLines(object):
    def test__denoise_lines(self):
        # 0) load input data
        lines_in = np.loadtxt(join(data_folder, 'test__denoise_lines__lines_in.txt'), delimiter=',')

        # 1) create default and runtime params
        default_params = DefaultParams()
        runtime_params = RuntimeParams()

        # 2) set runtime params
        runtime_params.H = 612
        runtime_params.W = 816
        runtime_params.SEGMENT_LENGTH_THRESHOLD = 22.0988

        # 3) run denoise_lines
        detections = denoise_lines(lines_in, default_params, runtime_params)

        print("OK")
