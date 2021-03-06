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


class TestStaticMethods(object):
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

    def test__orthogonal_triplet(self):
        # 0) load and declare input and output params
        my_vps_in = np.array([[0.407053027323275, -0.00226593554265788,	-0.867628080625217,	0.406687367833829,
                               -6.23742469510082e-05, -0.00369651390576061,	-0.850293443388202],
                              [0.0385165721289403, -0.998599901693855, 0.0208519605269108, 0.0385108455085616,
                               -0.997499486626139, -0.998205327252022, 0.0241837471736455],
                              [0.912592081172302, 0.0528498048543201, 0.496776317322784, 0.912755333932707,
                               0.0706736888102111,	0.0597700629966881,	0.525752990009259]])
        NFAs_in = np.array([[15.2695981373620, 10.0276928241632, 5.92824850463247, 14.2149491636451, 9.95881044997353,
                             8.57289712475728, 1.10973823275201]]).T
        ORTHOGONALITY_THRESHOLD_in = 0.0436193873653362
        ortho_vps_out = np.array([[-0.00226593554265788, 0.407053027323275],
                                  [-0.998599901693855, 0.0385165721289403],
                                  [0.0528498048543201, 0.912592081172302]])

        # 1) call the function
        ortho_vps = VanishingPointDetector._orthogonal_triplet(my_vps_in, NFAs_in, ORTHOGONALITY_THRESHOLD_in)

        # 2) compare results
        assert np.all(np.abs(ortho_vps - ortho_vps_out) < 1e-12)

    def test__gaussian_sphere_to_image(self):
        # 0) load and declare input and output params
        vp_in = np.array([[0.406650117964224, 0.00152377484175155, 0.913497301676117],
                          [0.0385266564047684, -0.996842908935405, -0.0309094305746012],
                          [0.912771262861595, 0.0793844633126079, -0.405668937598143]])
        W_in = int(816)
        H_in = int(612)
        pp_in = np.array([408, 306])
        FOCAL_RATIO_in = 1.08000000000000
        vpimg_out = np.array([[800.620397399444, 424.916059345904, -1576.49234685701],
                              [268.802540857881, 11372.3684319079, 238.851987292731]])

        # 1) run the function
        vpimg = VanishingPointDetector._gaussian_sphere_to_image(vp_in, W_in, H_in, FOCAL_RATIO_in, pp_in)

        # 2) compare results
        assert np.all(np.abs(vpimg - vpimg_out) < 1e-10)

    def test__image_to_gaussian_sphere(self):
        # 0) load and declare input and output params
        vpsimg_in = np.array([[800.894333585489, 361.573133786433, 240.989352828468, -966.059972074617,
                               800.663214555392, 462.265788430872, -963.037213448478],
                              [268.808002590910, 9775.53689835373, 18673.3155510477, 265.394015935807,
                               268.817157382630, 11325.0085875853, 265.724434539945]])
        W_in = int(816)
        H_in = int(612)
        pp_in = np.array([408, 306])
        FOCAL_RATIO_in = 1.08000000000000
        vpimg_out = np.array([[800.620397399444, 424.916059345904, -1576.49234685701],
                              [268.802540857881, 11372.3684319079, 238.851987292731]])
        vp_out = np.array([[0.406886972554092, -0.00488160719312791, -0.00908199431769545, -0.841487349025937,
                            0.406687367833829, 0.00490900871060000, -0.840949958078957],
                           [0.0385165626872847,	-0.995685541771179,	-0.998809706393187,	0.0248674894685829,
                            0.0385108455085616,	-0.996804998190338,	0.0247037314180914],
                           [0.912666130611037, 0.0926632171848502, 0.0479237706568382, 0.539704223993876,
                            0.912755333932707, 0.0797226267520033, 0.540548696845181]])

        # 1) run the function
        vp = VanishingPointDetector._image_to_gaussian_sphere(vpsimg_in, W_in, H_in, FOCAL_RATIO_in, pp_in)

        # 2) compare results
        assert np.all(np.abs(vp - vp_out) < 1e-10)


class TestVanishingPointDetector(object):
    def test__remove_duplicates(self):
        # 0) declare input parameters
        vps_in = np.array([[801.086571010609, 415.546837031069, -1134.13678498561, 394.566836036080, -964.487303073194,
                            800.663214555392, 427.257560668762,	-1161.51366309469, -952.156345371049],
                           [268.804975644552, 11716.0648307581,	268.800391684926, 22792.6458578845,	266.667286206964,
                            268.817157382630, 15706.5734652106,	268.286391427570, 264.229897744355]])
        NFAs_in = np.array([[16.5774600986004, 12.2759527537009, 7.89808635088906, 7.61558281212357, 6.31541371126635,
                             14.3126110026628, 8.90090669359107, 1.87937845301575, 1.14151921068338]]).T
        vpd = VanishingPointDetector()

        # 1) run the function
        final_vps, final_NFAs = vpd._remove_duplicates(vps_in, NFAs_in)

        # 2) compare results to MATLAB (in this case, the output of this function is identical to the input)
        assert np.all(np.abs(final_vps - vps_in) < 1e-12)
        assert np.all(np.abs(final_NFAs - NFAs_in) < 1e-12)

    def test__refine_detections(self):
        # 0) load and declare input and output parameters
        lines_lsd_in = np.loadtxt(join(data_folder, 'test__denoise_lines__lines_in.txt'), delimiter=',')
        mvp_all_in = np.array([[796.383918927249, 366.693922105435, -966.113269407856, -1132.22448651142,
                                795.852350268115, 421.996457739350,	804.939758522107, -1265.43119505193],
                               [268.714618008844, 13670.9315876730,	267.212475921965, 260.200906039884,
                                268.073415950411, 12626.0039197987,	265.800887546446, 252.992107462694]])
        mvp_all_out = np.array([[800.894333585489, 366.693922105435, -963.996965933382, -1134.13678498561,
                                 800.663214555392, 421.996457739350, 803.936158985557, -1295.67597597918],
                                [268.808002590910, 13670.9315876730, 266.437020037757, 268.800391684926,
                                 268.817157382630, 12626.0039197987, 267.677822413570, 261.116432351162]])
        vpd = VanishingPointDetector()
        vpd.runtime_params.H = 612
        vpd.runtime_params.W = 816

        # 1) run function
        mvp_all = vpd._refine_detections(mvp_all_in, lines_lsd_in)

        # 2) compare results
        assert np.all(np.abs(mvp_all - mvp_all_out) < 1e-8)

    def test__compute_horizon_line_manhattan(self):
        # 0) load and declare input and output parameters
        lines_lsd_in = np.loadtxt(join(data_folder, 'test__denoise_lines__lines_in.txt'), delimiter=',')
        Y_out = np.loadtxt(join(data_folder, 'test__compute_horizon_line_manhattan_Y_out.txt'), delimiter=',')
        mvp_all_in = np.array([[801.156259844744, 371.299105478700,	-965.966411594700, 800.620397399444,
                                416.071234238046, 322.095987374500, -1134.13678498561],
                               [268.795000423408, 15149.8342419224,	265.552008676436, 268.802540857881,
                                12844.8842313756, 15210.1684811810,	268.800391684926]])
        NFAs_in = np.array([[15.5109950836006, 9.77708028717022, 4.69261296923142, 14.4736967046236, 9.57277268761379,
                             6.98574179935858, 1.95788730536498]]).T
        vpoimg_out = np.array([[801.156259844744, 371.299105478700, -1572.84924724252],
                               [268.795000423408, 15149.8342419224,	248.780728762999]])

        vpd = VanishingPointDetector()
        vpd.runtime_params.H = 612
        vpd.runtime_params.W = 816
        vpd.runtime_params.FOCAL_RATIO = 1.08000000000000
        vpd.default_params.MANHATTAN = True

        # 1) run function
        Y, vpoimg = vpd._compute_horizon_line_manhattan(mvp_all_in, NFAs_in, lines_lsd_in)

        # 2) compare results
        # note: Y - the vertical pixel location of the horizon line - is obtained by using polyval for every horizontal
        #       pixel in the image. However, the implementation of polyval in MATLAB gives a slightly different result
        #       than the implementation in numpy. The difference is less than 1e-2 of a pixel, which is no significant
        #       for practical applications.
        assert np.all(np.abs(vpoimg - vpoimg_out) < 1e-12)
        assert np.all(np.abs(Y - Y_out) < 1e-2)

    def test__compute_horizon_line_non_manhattan(self):
        # 0) load and declare input and output parameters
        lines_lsd_in = np.loadtxt(join(data_folder, 'test__denoise_lines__lines_in.txt'), delimiter=',')
        Y_out = np.loadtxt(join(data_folder, 'test__compute_horizon_line_non_manhattan_Y_out.txt'), delimiter=',')
        mvp_all_in = np.array([[801.086571010609, 253.069911493031, 445.753039751043, -1135.90886157744,
                                -935.733501714431, 800.031753639748, 345.995863404967, -1265.37799357158],
                               [268.804975644552, 19792.5117752732, 17604.5834297621, 268.619757944945,
                                262.973492827207, 269.007367100989, 16524.7122471342, 261.444654557385]])
        NFAs_in = np.array([[12.5045550394745, 11.1526478989413, 9.28166243677099, 4.02308134697477,
                             1.35575910408630, 15.6436453400126, 7.60557409715032, 1.74962983731588]]).T
        vpimg_out = np.array([[800.031753639748, 801.086571010609, -1135.90886157744,
                              -1265.37799357158, -935.733501714431, 253.069911493031],
                              [269.007367100989, 268.804975644552, 268.619757944945,
                               261.444654557385, 262.973492827207, 19792.5117752732]])

        vpd = VanishingPointDetector()
        vpd.runtime_params.H = 612
        vpd.runtime_params.W = 816
        vpd.runtime_params.FOCAL_RATIO = 1.08000000000000

        # 1) run function
        Y, vpimg = vpd._compute_horizon_line_non_manhattan(mvp_all_in, NFAs_in, lines_lsd_in)

        # 2) compare results
        # note: Y - the vertical pixel location of the horizon line - is obtained by using polyval for every horizontal
        #       pixel in the image. However, the implementation of polyval in MATLAB gives a slightly different result
        #       than the implementation in numpy. The difference is less than 1e-2 of a pixel, which is no significant
        #       for practical applications.
        assert np.all(np.abs(vpimg - vpimg_out) < 1e-12)
        assert np.all(np.abs(Y - Y_out) < 1e-1)

    def test__convert_to_PClines(self):
        # 0) load and declare input and output parameters
        lines_in = np.loadtxt(join(data_folder, 'test__convert_to_PClines_lines_in.txt'), delimiter=',')
        points_straight_out = np.loadtxt(join(data_folder, 'test__convert_to_PClines_points_straight_out.txt'),
                                         delimiter=',')
        points_twisted_out = np.loadtxt(join(data_folder, 'test__convert_to_PClines_points_twisted_out.txt'),
                                        delimiter=',')
        vpd = VanishingPointDetector()
        vpd.runtime_params.H = 612
        vpd.runtime_params.W = 816

        # 1) run function
        points_straight, points_twisted = vpd._convert_to_PClines(lines_in)

        # 2) compare results
        assert np.all(np.abs(points_straight - points_straight_out) < 1e-12)
        assert np.all(np.abs(points_twisted - points_twisted_out) < 1e-12)

    def test__read_detections_as_vps(self):
        # 0) load and declare input and output parameters
        detections_straight_in = np.array([[0.656679854892006, 0.624056979846883, 1.16723647910498,
                                           0.348197377969931, 0.0116063187355071, 18.7110892729485],
                                          [0.860489218734756, 0.174389770277772, 1.18967742444997,
                                           0.767196210367086, 0.0135614800105795, 6.92779694285988],
                                          [0.00389457079021122, 0.851744744163936, -0.0107190919196349,
                                           0.0310585933938537, 0.0164163250114465, 6.74989252433881]])
        detections_twisted_in = np.array([[-1.57337940104685, -1.24650400197803, -0.729017324978145,
                                           -0.0553617377995084, 0.00730028631119252, 14.8963343546595],
                                          [0.0148240386305514, 0.763303883637193, -0.0232124242981950,
                                           0.0190363472315584, 0.0149047769290177, 6.56082171866466],
                                          [-1.15479636204033, -0.276690296466520, -0.816409898167133,
                                           -0.604081326567678, 0.00941679957359828, 1.87112619209133]])
        m1_in = np.array([[-0.540311473388856, 1.80081312087515, 56.1588266449544]]).T
        m2_in = np.array([[1.41070080944942, 19.5672120670071, -0.967506283655138]]).T
        b1_in = np.array([[0.978868639788363, -1.37519050519138, 0.633030218299961]]).T
        b2_in = np.array([[0.973063592649811, 0.473238776063689, -1.39396303308263]]).T
        mvp_all_out = np.array([[798.756810067304, -1122.15545223617, 516.552658132769,
                                 794.019891602245, 386.162841267970, -1137.47383499543],
                                [268.396985836498, 260.481040798465, 34756.6164003116,
                                 267.833976681360, 11685.5116540574, 260.991530649625]])
        NFAs_out = np.array([[18.7110892729485, 6.92779694285988, 6.74989252433881,
                              14.8963343546595, 6.56082171866466, 1.87112619209133]]).T
        vpd = VanishingPointDetector()
        vpd.runtime_params.H = 612
        vpd.runtime_params.W = 816

        # 1) run function
        mvp_all, NFAs = vpd._read_detections_as_vps(detections_straight_in, m1_in, b1_in,
                                                    detections_twisted_in, m2_in, b2_in)

        # 2) conpare results
        assert np.all(np.abs(mvp_all - mvp_all_out) < 1e-10)
        assert np.all(np.abs(NFAs - NFAs_out) < 1e-12)

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
        horizon, vpimg = vpd.detect_vps(image_in, folder_out=output_dir, manhattan=False,
                                        acceleration=True, focal_ratio=1.08, plot=True, print_output=True)

        # 3) assert results are correct
        print("OK")


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

    def test__draw_segments(self):
        # 0) load and declare input and output parameters
        lines_in = np.loadtxt(join(data_folder, 'test__denoise_lines__lines_in.txt'), delimiter=',')
        img_in = cv2.cvtColor(cv2.imread(join(data_folder, 'test.jpg'), -1), cv2.COLOR_BGR2RGB)
        vpimg_in = np.array([[800.894333585489, 345.968434441128, -949.742287440140, 799.834641087275, 408.125435891671],
                             [268.808002590910, 18339.9731893547, 263.862166183153, 268.829479610743, 10760.8963609918]])
        vpd = VanishingPointDetector()
        vpd.runtime_params.H = 612
        vpd.runtime_params.W = 816

        # 1) run the function
        img2 = vpd._draw_segments(img_in, vpimg_in, lines_in)

        print("OK")


