import os
import cv2
import numpy as np
from LSD import LSDWrapper
from matplotlib import pyplot as plt
from denoise_lines import denoise_lines
from PCLines import PCLines_straight_all
from find_detections import find_detections
from detect_vps_params import DefaultParams, RuntimeParams


class VanishingPointDetector:
    def __init__(self):
        # 1) declare default parameters
        self.default_params = DefaultParams()
        self.runtime_params = RuntimeParams()

        # 2) initialize wrappers
        self._lsd = LSDWrapper()

        # 3) initialize a placeholder for all figures
        self.figures = []

    def GetConfig(self, as_string: bool = False) -> dict or str:
        return self.params.GetConfig(as_string)

    def SetConfig(self, params_in: dict) -> None:
        self.params.SetConfig(params_in)

    def detect_vps(self,
                   img_in: str,
                   folder_out: str,
                   manhattan: bool,
                   acceleration: bool,
                   focal_ratio: float,
                   plot: bool = False,
                   print_output: bool = False) -> tuple:
        """
        Function for vanishing points and horizon line detection.
        :param img_in:           path to the input image.
        :param folder_out:       path to save resulting image and text files.
        :param manhattan:        boolean variable used to determine if the Manhattan-world hypothesis is assumed.
        :param acceleration:     boolean variable used to determine if acceleration using Figueiredo and Jain GMM
                                 algorithm should be used.
        :param focal_ratio:      ratio between the focal lenght and captor width
        :param plot:             TODO: understand and describe
        :param print_output:            TODO: understand and describe
        :return:
                - horizon_line: estimated horizon line in image coordinates
                - vpimg: esitmated vanishing points in image coordinates
        """
        pass
        # 0) Handle input
        assert isinstance(img_in, str) and os.path.isfile(img_in)
        assert isinstance(folder_out, str) and os.path.isdir(folder_out)
        assert isinstance(plot, bool) and isinstance(print_output, bool)

        # 1) read image (and update relevant runtime parameters)
        self.runtime_params.img_in = img_in
        img = cv2.imread(img_in, -1)
        self.runtime_params.H, self.runtime_params.W = img.shape[:2]

        # 2) calculate final LENGTH_THRESHOLD based on image size and LENGTH_THRESHOLD coefficients in default params
        self.runtime_params.SEGMENT_LENGTH_THRESHOLD = \
            np.sqrt(self.runtime_params.H + self.runtime_params.W) / self.default_params.SEGMENT_LENGTH_THRESHOLD_COEFF

        # 3) get line segments using LSD
        lines_lsd = self._lsd.run_lsd(img)

        # 4) denoise lines
        lines, line_endpoints = denoise_lines(lines_lsd, self.default_params, self.runtime_params)

        # 5) draw detected lines

        # 6) convert lines to PCLines
        points_straight, points_twisted = self._convert_to_PClines(lines)

        # 7) run alignment detector in straight and twisted spaces
        detections_straight, m1, b = find_detections(points_straight, self.default_params)
        detections_twisted, m2, b2 = find_detections(points_twisted, self.default_params)

        # 8) gather intial vanishing point detections (convert them from dual spaces)

        # 9) refine detections

        # 10) draw preliminary detections

        # 11) compute horizon line

        # 12) draw dual spaces

    def _draw_lines(self, lines1: np.ndarray, lines2: np.ndarray) -> None:  #TODO: finish
        H = self.runtime_params.H
        W = self.runtime_params.W
        LW = 1.5

        fig = plt.figure()
        ax2d = fig.add_subplots()

        for ii in range(len(lines1)):
            l = lines1[ii, :]
            ax2d.plot(l[::2], H - l[1::2], linewidth=LW)

        for ii in range(len(lines2)):
            l = lines2[ii, :]
            ax2d.plot(l[::2], H - l[1::2], 'r', linewidth=LW)

    def _convert_to_PClines(self, lines: np.ndarray) -> tuple:
        """
        Converts lines in the image to PCLines "straight" and "twisted" spaces
        :param lines: An Lx4 numpy array, where `L` is the number of lines.
        :return: A length-2 tuple that contains:
                 - points_straight: An Lx2 numpy array of points in the "straight" dual space,
                                    each point is the conversion of the corresponding line to said dual space.
                 - points_twisted: Same as above, but for the "twisted" dual space.
        """

        # 1) get runtime parameters
        H = self.runtime_params.H
        W = self.runtime_params.W
        L = len(lines)

        # 2) convert lines to points in both dual spaces
        points_straight = PCLines_straight_all(lines / np.tile([W, H], [L, 2]), 'straight')
        points_twisted = PCLines_straight_all(lines / np.tile([W, H], [L, 2]), 'twisted')

        # 3) impose boundaries of PClines space
        # Ihe CVPR paper uses a domain in the twisted space between [-2, 1]x[-2, 1]
        # Instead, this version uses [-2, 1]x[-1.5 1.5], which is more consistent with the transform (see IPOL article)
        z1 = points_straight[:, 0] > 2 or points_straight[:, 1] > 2 or points_straight[:, 0] < -1 or \
             points_straight[:, 1] < -1 or np.isnan(points_straight[:, 0]) or np.isnan(points_straight[:, 1])

        z2 = points_twisted[:, 0] > 1 or points_twisted[:, 1] > 1.5 or points_twisted[:, 0] < -2 or \
             points_twisted[:, 1] < -1.5 or np.isnan(points_twisted[:, 0]) or np.isnan(points_twisted[:, 1])

        return points_straight[z1], points_twisted[z2]

    def _read_detections_as_vps(self, detections_straight: np.ndarray, m1: np.ndarray, b1: np.ndarray,
                                detections_twisted: np.ndarray, m2: np.ndarray, b2: np.ndarray) -> tuple:
        """
        converts alignment detections to vanishing points in the image.
        returns mvp_all, a list of vanishing points and NFAs, their corresponding -log10(true NFA).
        TODO: complete doc
        :param detections_straight: A D1x6 numpy array (float).
        :param m1: A D1-length numpy array (float).
        :param b1: A D1-length numpy array (float).
        :param detections_twisted:
        :param m2:
        :param b2:
        :return:
        """
        # 0) get runtime parameters
        H = self.runtime_params.H
        W = self.runtime_params.W
        D1 = len(detections_straight)
        D2 = len(detections_twisted)

        # 1) check input
        if detections_straight.size == 0 and detections_twisted.size == 0:
            return np.empty((0, 2)), np.empty((0, 2))

        if not detections_straight.size == 0:
            NFAs1 = detections_straight[:, -1]
        else:
            NFAs1 = np.array([])

        if not detections_twisted.size == 0:
            NFAs2 = detections_twisted[:, -1]
        else:
            NFAs2 = np.array([])

        # 2) get vps in image coordinates (do PCLines inverse)
        d = 1
        x1 = b1
        y1 = d * m1 + b1

        x2 = b2
        y2 = -(-d * m2 + b2)

        x1 = x1 * W
        y1 = y1 * H

        x2 = x2 * W
        y2 = y2 * H

        vps1 = np.vstack([x1, y1])
        vps2 = np.vstack([x2, y2])

        mvp_all = np.hstack([vps1, vps2])
        NFAs = np.hstack([NFAs1, NFAs2])

        # 3) remove nan (infinity vp)
        z = np.isnan(mvp_all[0, :]) or np.isnan(mvp_all[1, :])

        return mvp_all[:, z], NFAs[z]

    def _refine_detections(self, mvp_all: np.ndarray, lines_lsd: np.ndarray) -> np.ndarray:
        """
        Refines VP detections using lines from LSD.
        :param mvp_all:
        :param lines_lsd:
        :return:
        """
        pass
