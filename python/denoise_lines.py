import numpy as np
from copy import deepcopy
import pyalignment_slow
import pyalignment_fast
from detect_vps_params import DefaultParams, RuntimeParams


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
    lengths = np.sqrt(np.sum(length_x_y ** 2, axis=1))[:, np.newaxis]

    # 2) find the angle of each line (in degrees)
    angles = np.rad2deg(np.arctan((lines[:, 3] - lines[:, 1]) / (lines[:, 2] - lines[:, 0])))[:, np.newaxis]

    # 3) dividing the set of lines into short and long lines
    # 3.1) find indices of long lines, which do not need denoising
    z_large = lengths > runtime_params.SEGMENT_LENGTH_THRESHOLD

    # 3.2) find indices of short lines, which need denoising
    z_short = np.bitwise_not(z_large)

    # 4) group lines according to orientation
    all_detections = np.empty((0, 4), dtype=np.float64)
    for ANG in np.array(range(0, 180, 30)).astype(np.float64):
        # 4.1) find which line has an angle which is 20 degreed from current orientation
        # 4.1.1) subtract angle of orientation
        angdiff = np.abs(angles - ANG)

        # 4.1.2) TODO
        angdiff = np.min(np.stack([angdiff, np.abs(angdiff - 180.)]), axis=0)

        # 4.1.3) mark relevant lines
        z_20_deg_diff = angdiff < 20.

        # 4.2) creating groups of lines according to orientation and finding alignments of line segment endpoints
        # 4.2.1) find large lines with angles less than 20 degrees from currect orientation
        z_large_20 = np.bitwise_and(z_large, z_20_deg_diff)

        # 4.2.2) find large lines with angles less than 20 degrees from currect orientation
        z_short_20 = np.bitwise_and(z_short, z_20_deg_diff)

        # 4.3) save angle of orientation to runtime parameters in order to pass it down the call stack
        runtime_params.ANG = ANG

        # 4.4) detect alignments of short lines
        if len(z_short_20) > 0:
            detections = find_endpoints_detections(lines[z_short_20], default_params, runtime_params)
            all_detections = np.append(all_detections, detections, axis=0)

        # 4.5) detect alignments of large lines
        if len(z_large_20) > 0:
            detections = find_endpoints_detections(lines[z_large_20], default_params, runtime_params)
            all_detections = np.append(all_detections, detections, axis=0)

    # 5) TODO
    lines_short_denoised = deepcopy(all_detections)


def find_endpoints_detections(lines, default_params, runtime_params):
    # convert lines to point segments (y axis inverted)
    H = runtime_params.H
    point_segments = np.hstack([lines[:, 0], lines[:, 2], H - lines[:, 1], H - lines[:, 2]])

    # TODO: check wtf is params.endpoints in MATLAB

    # decide if acceleration is needed
    USE_ACCELERATION = False
    if len(point_segments) > default_params.MAX_POINTS_ACCELERATION and default_params.ACCELERATION:
        print('- USE ACCELERATED VERSION -')
        USE_ACCELERATION = True

    # normalize segment endpoints
    M = np.max(point_segments)
    m = np.min(point_segments)

    # this version of the alignment detector expects a 512x512 domain
    point_segments_normed = ((point_segments - m) / (M - m)) * 512.

    # run alignment detections
    if USE_ACCELERATION:
        candidates = run_mixtures(point_segments_normed, default_params.GMM_Ks, '')
        segment_detections = pyalignment_fast.detect_alignments_fast(point_segments_normed)
    else:
        segment_detections = pyalignment_slow.detect_alignments_slow(point_segments_normed)

    # convert detections to line coordinates
    line_detections = (segment_detections / 512.) * (M - m) + m

    return line_detections


