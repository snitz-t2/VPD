import sys
import cv2
import pytest
import shutil
import numpy as np
from os.path import join, abspath, dirname, isdir
from os import makedirs

my_dir = dirname(abspath(__file__))
data_folder = join(my_dir, 'input_data')
sys.path.append(dirname(my_dir))

from detect_vps import VanishingPointDetector


class TestVanishingPointDetector(object):
    def test__line_to_homogeneous(self):
        # 0) load input data and output data
        lines_in = np.loadtxt(join(data_folder, 'test__denoise_lines__lines_in.txt'), delimiter=',')
        lines_homog_out = np.loadtxt(join(data_folder, 'test__line_to_homogeneous__lines_homog_out.txt'), delimiter=',')

        # 1) call the function
        lines_homog = VanishingPointDetector._line_to_homogeneous(lines_in)

        # 2) assert correctness
        assert np.all(np.abs(lines_homog - lines_homog_out) < 1e-9)

    def test__aggclus(self):
        # 0) load input data and output data
        X_in = np.array([[800.894333585489,268.80800259091],
                         [345.968434441128,18339.9731893547],
                         [-949.74228744014,263.862166183153],
                         [799.834641087275,268.829479610743],
                         [408.125435891671,10760.8963609918]])
        clus_out = [[0, 3], [1], [2], [4]]
        thresh_in = 1e-2

        # 1) call the function
        clus = VanishingPointDetector._aggclus(X_in, thresh_in)

        # 2) assert correctness
        assert len(clus) == len(clus_out)
        for ii in range(len(clus)):
            assert len(clus[ii]) == len(clus_out[ii])
            for jj in range(len(clus_out[ii])):
                assert clus[ii][jj] == clus_out[ii][jj]

    def test__detect_vps(self):

        # 0) define input data (image)
        image_in = join(data_folder, 'test.jpg')

        # 1) create the output folder (delete if exist)
        output_dir = join(my_dir, 'output_data')
        if isdir(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)
        makedirs(output_dir)

        # 1) initiate the class
        vpd = VanishingPointDetector()

        # 2) run the algorithm
        horizon = vpd.detect_vps(image_in, folder_out=output_dir, manhattan=False,
                                 acceleration=True, focal_ratio=1.08, plot=True, print_output=True)

        # 3) assert results are correct


class TestVanishingPointDetectorDrawingFunctions(object):
    def test__drawline(self):
        # 0) load output files and image
        BW_out = cv2.imread(join(data_folder, 'test__drawline_BW_out.tif'), -1)
        ind_out = np.loadtxt(join(data_folder, 'test__drawline_ind_out.txt'), delimiter=',')
        label_out = np.loadtxt(join(data_folder, 'test__drawline_label_out.txt'), delimiter=',')

        # 1) convert output arrays from MATLAB indices to python indices
        ind_out = ind_out.astype(int) - 1
        label_out = label_out.astype(int) - 1

        # 1) generate input data
        p1 = np.array([[9, 9], [22, 99], [-15, -41]])
        p2 = np.array([[49, 49], [89, 99], [49, 49]])
        ind, label = VanishingPointDetector._drawline(p1, p2, *BW_out.shape)

        # 2) assert results
        assert np.all(ind == ind_out) and np.all(label == label_out)

        # 3) verify on image
        BW = np.zeros(BW_out.shape, dtype=np.uint8).ravel(order='F')
        BW[ind] = 255
        BW = BW.reshape(BW_out.shape, order='F')
        assert np.all(BW == BW_out)


