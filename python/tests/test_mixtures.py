import sys
import cv2
import pytest
import numpy as np
from os.path import join, abspath, dirname

my_dir = dirname(abspath(__file__))
data_folder = join(my_dir, 'input_data')
sys.path.append(dirname(my_dir))

from mixtures import *


class TestMixtures(object):
    def test__my_multinorm(self):
        # 0) load input and output data
        data = np.load(join(data_folder, 'test__my_multinorm_data.npz'))

        # 1) run function
        y = my_multinorm(data['x_in'], data['m_in'], data['covar_in'], data['npoints_in'])

        # 2) evaluate results are correct
        assert np.all(np.abs(y - data['y_out']) < 1e-12)

    def test__get_ellipse_endpoints(self):
        # 0) define input and output parameters for the function
        m_in = np.array([497.222307017388, 63.7813846025081])
        cov_in = np.array([[153.194914856023, -122.044037769752],
                           [-122.044037769752, 482.881877534115]])
        level_in = 2
        pair_out = np.array([482.891091980472, 107.223189816438, 511.553522054304, 20.3395793885783])

        # 1) call the function
        pairs = get_ellipse_endpoints(m_in, cov_in, level_in)

        # 2) evaluate results are correct
        assert np.all(np.abs(pairs - pair_out) < 1e-12)

    def test__mixtures4(self):
        # 0) load input data
        K = 30
        points_in = np.loadtxt(join(data_folder, 'test__mixtures__points_in.txt'), delimiter=',')
        randindex_in = np.loadtxt(join(data_folder, 'test__mixtures4__randindex_in.txt'), delimiter=',')


        # 1) load output data
        bestK_out = 19
        bestpp_out = np.array([0.0549064880309287,	0.110571043475691,	0.0207546638782990,	0.0957239608207365,
                               0.0353552497626214,	0.0246667732058497,	0.0385793606025974,	0.112229882179203,
                               0.102517249099987,	0.0192207901688042,	0.0672670763619231,	0.0474814918320450,
                               0.0492191284329155,	0.0879442815786799,	0.0338374521834145,	0.0223504140054997,
                               0.0405698709696434,	0.0299971196877796,	0.00680770372338056])
        dl_out = np.array([5488.56514325243,    5248.95523793431,	5181.10171353679,	5148.00226392782,
                           5115.13286040284,	5086.70884079815,	5070.61531841617,	5060.03958353263,
                           5044.02516905918,	5030.70829814361,	5021.92918344284,	5018.71113285890,
                           5015.53392049489,	5011.00299832735,	5006.01121382907,	5001.47211788202,
                           4994.92139488937,	4992.61569104290,	4989.66694748293,	4985.52172973777])
        countf_out = 19
        bestmu_out = np.loadtxt(join(data_folder, 'test__mixtures4__bestmu_out.txt'), delimiter=',')
        bestcov_out = np.loadtxt(join(data_folder, 'test__mixtures4__bestcov_out.txt'), delimiter=',')
        bestpairs_out = np.loadtxt(join(data_folder, 'test__mixtures4__bestpairs_out.txt'), delimiter=',')

        # 1) process data
        # 1.1) cheese unique points
        unique_points = np.unique(np.round(points_in).astype(float), axis=0)

        # 1.2) convert random index from MATLAB to python
        randindex_in = (randindex_in - 1).astype(int)

        # 1.3) convert bestcov_out from MATLAB to python
        bestcov_out_corrected = np.zeros((2, 2, bestcov_out.shape[1] // 2))
        for ii in range(bestcov_out.shape[1] // 2):
            bestcov_out_corrected[:, :, ii] = bestcov_out[:, 2*ii:2*ii+2]

        # 2) call the function
        bestK, bestpp, bestmu, bestcov, dl, countf, bestpairs = mixtures4(unique_points.T,
                                                                          kmin=max(2, K-7), kmax=K, regularize=0.,
                                                                          th=1e-4, covoption=0,
                                                                          randindex=randindex_in)

        # 3) evaluate results are correct
        assert bestK == bestK_out
        assert countf == countf_out
        assert np.all(np.abs(bestpp - bestpp_out) < 1e-10)
        assert np.all(np.abs(bestmu - bestmu_out) < 1e-10)
        assert np.all(np.abs(bestcov - bestcov_out_corrected) < 1e-8)
        assert len(dl) == len(dl_out)
        for idx, val in enumerate(dl_out):
            assert abs(dl[idx] - val) < 1e-10
        assert np.all(np.abs(bestpairs - bestpairs_out) < 1e-10)

    def test__run_mixtures(self):
        # 0) load input data
        points_in = np.loadtxt(join(data_folder, 'test__mixtures__points_in.txt'), delimiter=',')

        # 1) run mixtures algorithm
        GMM_Ks = np.array([30, 30, 30], dtype=np.int8)
        all_bestpairs = run_mixtures(points_in, GMM_Ks)

        # 2) check returned shape validity (since `mixtures4` uses a random generator, it is hard to validate
        #    `run_mixtures`, which is supposed to use it multiple times with different random permutations).
        assert all_bestpairs.shape[1] == 2

        print("OK")



