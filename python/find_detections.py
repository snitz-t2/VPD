import numpy as np
import pyalignments_slow
import pyalignments_fast
from mixtures import run_mixtures
from detect_vps_params import DefaultParams


def find_detections(points: np.ndarray, default_params: DefaultParams) -> tuple:
    """
    TODO: finish description of line alignment detector (from the paper)
    Detect line segments from a group of points.
    Returns the endpoints, as well as the incline `m` and intercept `b` parameters for each detected line.
    :param points: An nx2 numpy array, containing the line segment endpoints [xi, yi].
    :param default_params: An instance of the `DefaultParams` class, containing the configured parameters of
                           the vanishing points detection algorithm.
    :return: A length-3 tuple with the following outputs:
             - detections: An Lx6 numpy array, where `L` stands for the number of aligned lines detected by the chosen
                           implementation of the alignment detector.
             - m, b:  An Lx1 numpy arrays that stand for the slope and the intercept of each detected line.
    """
    # decide if acceleration is needed
    USE_ACCELERATION = False
    if len(points) > default_params.MAX_POINTS_ACCELERATION and default_params.ACCELERATION:
        print('- USE ACCELERATED VERSION -')
        USE_ACCELERATION = True

    # normalize segment endpoints
    M = np.max(points)
    m = np.min(points)

    # this version of the alignment detector expects a 512x512 domain
    point_segments_normed = ((points - m) / (M - m)) * 512.

    # run alignment detections
    if USE_ACCELERATION:
        candidates = run_mixtures(point_segments_normed, default_params.GMM_Ks)
        detections = pyalignments_fast.detect_alignments_fast(point_segments_normed, candidates)
    else:
        detections = pyalignments_slow.detect_alignments_slow(point_segments_normed)

    if detections.size > 0:
        # convert detections to line coordinates
        dets = detections[:, :4]
        dets = dets / 512. * (M - m) + m
        detections[:, :4] = dets
        detections[:, 4] = detections[:, 4] / 512. * (M - m) + m

        # get parameters of detected lines
        x1 = dets[:, 0]; y1 = dets[:, 1]; x2 = dets[:, 2]; y2 = dets[:, 3]
        dy = y2 - y1
        dx = x2 - x1
        m = dy / dx
        b = (y1 * x2 - y2 * x1) / dx    # This may result in Nan or Inf, but this is handled in `read_detections_as_vps`
    else:
        m = np.array([])
        b = np.array([])

    return detections, m, b
