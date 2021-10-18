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
        lines_out, lines_short_denoised = denoise_lines(lines_in, default_params, runtime_params)

        # 4) since the evalueated function uses a random process, we cannot evaluate the numbers of the returned lines.
        #    however, we CAN run the function multiple times and get a minimal number of detected lines.
        #    also, `lines_out` contain `lines_short_denoised`, and we know there are number of large lines that
        #    the function does not discard, so `lines_out` must be larger than `lines_short_denoised`.
        assert len(lines_short_denoised) > 10
        assert len(lines_out) > len(lines_short_denoised)

        print("OK")
