import numpy as np


def PCLines_straight_all(xy_in: np.ndarray, space: str, d: float = 1.) -> (np.ndarray, np.ndarray):
    """
    This function converts lines and points from the image space into two possible dual spaces:
    - straight
    - twisted
    more information can be found in the article:
                                       `Vanishing Point Detection in Urban Scenes Using Point Alignments`, section 2.3.

    :param xy_in: A numpy array, with size nx4 (lines) or nx2 (points), where n is the number of lines/points.
                  Each line representation is [x1, y1, x2, y2] in image space.
                  Each point representation is [x, y] in image space.
    :param space: str. Can be only `straight` or `twisted`. Represents the name of the dual space that the points/lines
                  will be converted into.
    :param d: float. aribitrary distance between vertical axes x and y. default is 1.0, change is not recommended.
    :return: (u, v) coordinates of converted lines/points in specified dual space (each coordinate is a numpy array,
             which length is the number of points/lines).
    """
    # 0) handle input
    space_types = ('straight', 'twisted')
    if not (isinstance(xy_in, np.ndarray) and xy_in.shape[0] > 0 and xy_in.shape[1] in (2, 4)):
        raise ValueError("`xy_in` is incorrect type or shape, see documentation.")
    if not (isinstance(space, str) and space in space_types):
        raise ValueError("`space` must be one of the following strings: `straight` or `twisted`.")

    # 1) In case `xy_in` are lines
    if xy_in.shape[1] == 4:
        dx = xy_in[:, 2] - xy_in[:, 0]  # x2 - x1
        dy = xy_in[:, 3] - xy_in[:, 1]  # y2 - y1

        m = dy / dx
        b = (xy_in[:, 1] * xy_in[:, 2] - xy_in[:, 3] * xy_in[:, 0]) / dx  # (y1.*x2 - y2.*x1)/dx

        # homogeneous coordinates
        if space == 'straight':
            pc_line = np.hstack([d * np.ones((len(b), 1)), b[:, np.newaxis], (1. - m)[:, np.newaxis]])

        else:   # twisted
            pc_line = np.hstack([-d * np.ones((len(b), 1)), -b[:, np.newaxis], (1. + m)[:, np.newaxis]])

        u = pc_line[:, 0] / pc_line[:, 2]
        v = pc_line[:, 1] / pc_line[:, 2]

    # 2) In case `xy_in` are points
    else:
        if space == 'straight':
            m = (xy_in[:, 1] - xy_in[:, 0]) / d          # (y - x) / d
        else:   # twisted
            m = (xy_in[:, 1] + xy_in[:, 0]) / d          # (y - x) / d

        u = m
        v = xy_in[:, 0]

    return u, v
