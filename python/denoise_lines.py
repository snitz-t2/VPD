import numpy as np
from .detect_vps_params import DefaultParams, RuntimeParams


def denoise_lines(lines, default_params, runtime_params):
    """
    This function takes a list of line segments and performs denoising by:
    - dividing the set of lines into short and long lines
    - creating groups of lines according to orientation and finding alignments of line segment endpoints
    - adding the alignments found to the list of lines
    - removing short lines from the list
    :param lines:
    :param default_params:
    :param runtime_params:
    :return:
    """
    # 0) handle input
    assert isinstance(lines, np.ndarray) and len(lines.shape) == 2 and lines.shape[1] == 4
    assert isinstance(default_params, DefaultParams)
    assert isinstance(runtime_params, RuntimeParams)

    # 1) calculate the length of each line (sqrt(dx^2 + dy^2))
    length_x_y = np.vstack([lines[:, 2] - lines[:, 0], lines[:, 3] - lines[:, 1]]).T
    lengths = np.sqrt(np.sum(length_x_y ** 2, axis=1), axis=1)

    # 2) find the angle of each line (in degrees)
    angles = np.rad2deg(np.arctan(lines[:, 3] - lines[:, 1] / lines[:, 2] - lines[:, 0]))

    # 3) find long lines, which do not need denoising
    z = lengths < runtime_params.SEGMENT_LENGTH_THRESHOLD
    large_lines = lines[z]

    # 3) find short lines, which need denoising
    short_lines = lines[np.bitwise_not(z)]

    # 5) group lines according to orientation
    for ANG in range(0, 180, 30):
        pass # TODO: finishi after wrapping endpoint detection
