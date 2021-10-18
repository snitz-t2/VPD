import os
import cv2
import numpy as np
from LSD import LSDWrapper
from denoise_lines import denoise_lines
from detect_vps_params import DefaultParams, RuntimeParams


class VanishingPointDetector:
    def __init__(self):
        # 1) declare default parameters
        self.default_params = DefaultParams()
        self.runtime_params = RuntimeParams()

        # 2) initialize wrappers
        self._lsd = LSDWrapper()

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
        self.runtime_params.H, self.runtime_params.W = img.shape

        # 2) calculate final LENGTH_THRESHOLD based on image size and LENGTH_THRESHOLD coefficients in default params
        self.runtime_params.SEGMENT_LENGTH_THRESHOLD = \
            np.sqrt(self.runtime_params.H + self.runtime_params.W) / self.default_params.SEGMENT_LENGTH_THRESHOLD_COEFF

        # 3) get line segments using LSD
        lines_lsd = self._lsd.run_lsd(img)

        # 4) denoise lines
        lines, line_endpoints = denoise_lines(lines_lsd, self.default_params, self.runtime_params)


