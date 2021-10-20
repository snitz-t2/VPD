import os
import cv2
import numpy as np
from copy import deepcopy
from LSD import LSDWrapper
from matplotlib import pyplot as plt
from denoise_lines import denoise_lines
from PCLines import PCLines_straight_all
from find_detections import find_detections
from detect_vps_params import DefaultParams, RuntimeParams
from scipy.linalg import null_space
from scipy.spatial.distance import cdist


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
        :param focal_ratio:      ratio between the focal length and captor width
        :param plot:             TODO: understand and describe
        :param print_output:            TODO: understand and describe
        :return:
                - horizon_line: estimated horizon line in image coordinates
                - vpimg: esitmated vanishing points in image coordinates
        """
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
        if self.runtime_params.plot_graphs:
            self._draw_lines(lines, line_endpoints)

        # 6) convert lines to PCLines
        points_straight, points_twisted = self._convert_to_PClines(lines)

        # 7) run alignment detector in straight and twisted spaces
        detections_straight, m1, b1 = find_detections(points_straight, self.default_params)
        detections_twisted, m2, b2 = find_detections(points_twisted, self.default_params)

        # 8) gather intial vanishing point detections (convert them from dual spaces)
        mvp_all, NFAs = self._read_detections_as_vps(detections_straight, m1, b1,
                                                     detections_twisted, m2, b2)

        # 9) refine detections
        mvp_all = self._refine_detections(mvp_all, lines_lsd)

        # 10) draw preliminary detections
        mvp_all, NFAs = self._remove_duplicates(mvp_all, NFAs)

        # 11) draw preliminary detections
        if self.runtime_params.plot_graphs:
            img2 = self._draw_segments(img, mvp_all, lines_lsd)
            cv2.imwrite(os.path.join(self.runtime_params.folder_out, 'vps_raw.png'), img2)

        # 12) compute horizon line
        if mvp_all.size > 0:
            if self.default_params.MANHATTAN:
                horizon_line, vpimg = self._compute_horizon_line_manhattan(mvp_all, NFAs, lines_lsd)
            else:
                horizon_line, vpimg = self._compute_horizon_line_non_manhattan(mvp_all, NFAs, lines_lsd)
        else:
            horizon_line = np.array([])
            vpimg = np.array([])
            print("No vanishing points found")

        # 13) draw dual spaces
        if self.runtime_params.plot_graphs:
            self._draw_dual_spaces(points_straight, detections_straight, 'straight', vpimg)
            self._draw_dual_spaces(points_twisted, detections_twisted, 'twisted', vpimg)

        # 14) finish algorithm
        print("Finished")
        return horizon_line, vpimg

    def _draw_lines(self, lines1: np.ndarray, lines2: np.ndarray) -> None:
        raise NotImplementedError
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

    def _draw_segments(self, img: np.ndarray, mvp_all: np.ndarray, lines_lsd: np.ndarray) -> None:
        raise NotImplementedError

    def _draw_dual_spaces(self, points, detections, space, vpimg):
        raise NotImplementedError

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
        D = mvp_all.shape[1]
        mvp_refined = np.zeros((D, 2))
        for ii in range(D):
            vp = mvp_all[:, ii].T
            vp = self._refine_vp(lines_lsd, vp)
            mvp_refined[ii, :] = vp

        return mvp_refined

    def _refine_vp(self, lines, vp):
        """
        Given a cluster of line segments, and two segments indicated by p1 and p2,
        obtain the main vanishing point determined by the segments
        :param lines:
        :param vp:
        :return:
        """
        # 0) get configured parameters from class
        THRESHOLD = self.default_params.REFINE_THRESHOLD
        H = self.runtime_params.H
        W = self.runtime_params.W

        # 1) copy the original vanishing point
        vp_orig = deepcopy(vp)

        # -------------------------- beginning of `refine_vp_iteration` function in MATLAB -----------------------------

        # 2) find intersection of each line in cluster with vanishing point segments

        mp = np.vstack([lines[:, 0] + lines[:, 2], lines[:, 1] + lines[:, 3]]).T / 2

        L = lines.shape[0]
        O = np.ones((L, 1))
        vpmat = np.tile(vp, [L, 1])

        VP = np.cross(np.hstack([mp, O]).T, np.hstack([vpmat, O]).T).T

        VP3 = np.tile(VP[:, 2], [1, 3])
        VP = VP / VP3

        a = VP[:, 0]
        b = VP[:, 1]

        # 3) get angle between lines
        angle = np.abs(np.arctan(-a / b) - np.arctan((lines[:, 3] - lines[:, 1]) / (lines[:, 2] - lines[:, 0])))
        angle = np.min(angle, np.pi - angle)
        z2 = angle < np.deg2rad(THRESHOLD)

        # 4) obtain a refined VP estimate from sub-cluster z2
        lengths = np.sum((lines[:, 2] - lines[:, 2:]) ** 2, axis=1)
        weights = lengths / np.max(lengths)
        lis = self._line_to_homogeneous(lines)

        Is = np.diag([1, 1, 0])

        l2 = lis[z2, :].T
        w2 = weights[z2]
        w2 = np.tile(w2, [1, 3]).T

        b = np.dot(Is.T @ l2, l2).T
        b = np.tile(b, [1, 3]).T
        Q = (w2 * l2 / b) @ l2.T

        p = np.array([[0, 0, 1]]).T
        A = np.hstack([2 * Q, -p])

        vp = null_space(A)

        vp = (vp[:2, 0] / vp[2, 1]).T

        # -------------------------- end of `refine_vp_iteration` function in MATLAB -----------------------------------

        # 5) check how much the refined vp changed relatively to the original vp
        variation = np.linalg.norm(vp - vp_orig) / np.linalg.norm(vp_orig)

        if variation > self.default_params.VARIATION_THRESHOLD:
            return vp_orig                       # vp changed too much, probably unstable conditions, keeping initial vp
        else:
            return vp

    def _remove_duplicates(self, vps, NFAs):
        """
        Identifies and removes duplicate detections, keeping only most significant ones.
        :param vps:
        :param NFAs:
        :return:
        """
        THRESHOLD = self.default_params.DUPLICATES_THRESHOLD
        clus = self._aggclus(vps.T, THRESHOLD)

        final_vps = np.empty((2, 0))
        final_NFAs = np.array([])

        for ii in range(len(clus)):
            c = clus[ii]
            if len(c) == 1:
                final_vps = np.append(final_vps, vps[:, c], axis=1)
                final_NFAs = np.append(final_NFAs, NFAs[c])
            else:
                I = np.argmax(NFAs[c])
                final_vps = np.append(final_vps, vps[:, c[I]], axis=1)
                final_NFAs = np.append(final_NFAs, NFAs[c[I]])

        return final_vps, final_NFAs[:, np.newaxis]

    def _compute_horizon_line_manhattan(self, mvp_all, NFAs, lines_lsd):
        """
        computes horizontal line from vps and using the NFA values to apply orthogonality constraints.
        saves data to output image and output text file.
        :param mvp_all:
        :param NFAs:
        :param lines_lsd:
        :return:
        """
        raise NotImplementedError
        H = self.runtime_params.H
        W = self.runtime_params.W

        pp = np.array([W, H]) / self.default_params.ppd
        FOCAL_RATIO = self.runtime_params.FOCAL_RATIO

        my_vps = self._image_to_gaussian_sphere(mvp_all, W, H, FOCAL_RATIO, pp)

        my_vps[np.isnan(my_vps)] = 1

        # impose orthogonality
        my_orthogonal_vps = orthogonal_triplet(my_vps, NFAs, self.default_params.ORTHOGONALITY_THRESHOLD)

    def _compute_horizon_line_non_manhattan(self, mvp_all, NFAs, lines_lsd):
        """
        Computes the horizon line when the Manhattan-world hypothesis cannot be assumed.
        :param mvp_all:
        :param NFAs:
        :param lines_lsd:
        :return:
        """
        raise NotImplementedError


    @staticmethod
    def _aggclus(X: np.ndarray, THRESHOLD: float) -> list:
        """
        agglomerative clustering using single link.
        TODO: finish doc
        """
        N = X.shape[0]

        D = cdist(X, X)

        n = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)[:, np.newaxis]
        n = np.tile(n, [1, N])
        n = np.maximum(n, n.T)      # max norm

        D /= n
        np.fill_diagonal(D, np.inf)

        clus = [ [] for _ in range(N)]
        for ii in range(N):
            clus[ii] = [ii]

        D_flat = D.flatten(order='F')
        I = np.argmin(D_flat)
        V = D_flat[I]

        while V < THRESHOLD:
            ii, jj = np.unravel_index(3, D.shape, order='F')

            if ii > jj:
                t = ii
                ii = jj
                jj = t

            clus[ii] += clus[jj]
            clus.pop(jj)

            d = np.minimum(D[ii, :], D[jj, :])

            D[ii, :] = d
            D[:, ii] = d.T
            D = np.delete(D, obj=jj, axis=0)
            D = np.delete(D, obj=jj, axis=1)

            D[ii, ii] = np.inf

            D_flat = D.flatten(order='F')
            I = np.argmin(D_flat)
            V = D_flat[I]

        return clus

    @staticmethod
    def _line_to_homogeneous(l: np.ndarray) -> np.ndarray:
        """
        converts lines in x1 y1 x2 y2 format to homogeneous coordinates.
        :param l:
        :return:
        """
        x1 = l[:, 0]
        y1 = l[:, 1]
        x2 = l[:, 2]
        y2 = l[:, 3]

        dx = x2 - x1
        dy = y2 - y1
        a = -dy
        b = dx
        c = -a * x1 -b * y1

        L = np.vstack([a, b, c]).T
        return L

    @staticmethod
    def _image_to_gaussian_sphere(vpsimg: np.ndarray, W: int, H: int,
                                  FOCAL_RATIO: float, pp: np.ndarray) -> np.ndarray:
        vp = np.vstack([vpsimg[0, :] - pp[0],
                       (H - vpsimg[1, :]) - (H - pp[1]),
                       np.ones(vpsimg[1, :].shape) * W * FOCAL_RATIO])

        vp /= np.tile(np.sqrt(np.sum(vp ** 2)), [3, 1])

        return vp

    @staticmethod
    def _orthogonal_triplet(my_vps: np.ndarray, NFAs: np.ndarray, ORTHOGONALITY_THRESHOLD: float):
        """
        Returns most significant orthogonal triplet.
        Identifies and removes duplicate detections, keeping only most significant ones.
        :param my_vps:
        :param NFAs:
        :param ORTHOGONALITY_THRESHOLD:
        :return:
        """
        N = my_vps.shape[1]
        nfa_scores_triplets = np.array([])
        ortho_scores_triplets = np.array([])
        triplets = np.array([])
        nfa_scores_pairs = np.array([])
        ortho_scores_pairs = np.array([])
        pairs = np.array([])

        for ii in range(N):
            vpi = my_vps[:, ii]
            nfai = NFAs[ii]

            for jj in range(ii-1):
                if ii == jj:
                    continue
                vpj = my_vps[:, jj]
                nfaj = NFAs[jj]
                scoreij = np.abs(np.dot(vpi, vpj))

                nfa_scores_pairs = np.append(nfa_scores_pairs, nfai + nfaj)
                ortho_scores_pairs = np.append(ortho_scores_pairs, scoreij)

                pairs = np.append(pairs, [ii, jj])

                for kk in range(N):
                    if kk == jj or jj == ii:
                        continue
                    vpk = my_vps[:, kk]
                    nfak = NFAs[kk]
                    scorejk = np.abs(np.dot(vpj, vpk))
                    scoreik = np.abs(np.dot(vpi, vpk))

                    # get orthogonality score
                    ortho_score = np.max([scoreij, scorejk, scoreik])
                    nfa_score = nfai + nfaj + nfak

                    nfa_scores_triplets = np.append(nfa_scores_triplets, nfa_score)
                    ortho_scores_triplets = np.append(ortho_scores_triplets, ortho_score)
                    triplets = np.append(triplets, [ii, jj, kk])

        ORTHO_SCORE_THRESHOLD_TRIPLETS = ORTHOGONALITY_THRESHOLD
        ORTHO_SCORE_THRESHOLD_pairS = ORTHOGONALITY_THRESHOLD

        z3 = ortho_scores_triplets <= ORTHO_SCORE_THRESHOLD_TRIPLETS
        nfa_scores_triplets = nfa_scores_triplets[z3]
        ortho_scores_triplets = ortho_scores_triplets[z3]
        triplets = triplets[z3, :]

        z2 = ortho_scores_pairs <= ORTHO_SCORE_THRESHOLD_pairS
        nfa_scores_pairs_orig = deepcopy(nfa_scores_pairs)
        ortho_scores_pairs_orig = deepcopy(ortho_scores_pairs)
        pairs_orig = deepcopy(pairs)
        nfa_scores_pairs = nfa_scores_pairs[z2]
        ortho_scores_pairs = ortho_scores_pairs[z2]
        pairs = pairs[z2, :]

        if triplets.size == 0:  # no orthogonal triplets, return pair
            if pairs.size ==0:  # no triplets or pairs, get the most orthogonal pair
                I = np.argsort(ortho_scores_pairs_orig)
                pair = pairs_orig[I[0], :]
                ortho_vps = np.array([my_vps[:, pair[0]], my_vps[:, pair[1]]])
            else:               # by nfa
                I = np.argsort(nfa_scores_pairs)
                pair = pairs[I[-1], :]
                ortho_vps = np.array([my_vps[:, pair[0]], my_vps[:, pair[1]]])
        else:
            I = np.sort(nfa_scores_triplets)
            triplet = triplets[I[-1], :]
            ortho_vps = np.array([my_vps[:, triplet[0]], my_vps[:, triplet[1]], my_vps[:, triplet[2]]])

        return ortho_vps



    @staticmethod
    def _drawline(p1: np.ndarray, p2: np.ndarray, M: int, N: int) -> tuple:
        """
        DRAWLINE Returns the geometric space (matrix indices) occupied by a line segment in a MxN matrix.
        Each line segment is defined by two endpoints.
        :param p1: set of endpoints (Px2 numpy array). [x1 y1]
        :param p2: set of endpoints connect to p1 (Px2 numpy array). [x2 y2]
        :param M: Image height (number of rows).
        :param N: Image width (number of columns).
        :return: a length-2 tuple:
                 - IND - matrix indices occupied by the line segments.
                 - LABEL - label tag of each line drawn (from 1 to P).
        """
        # 0) handle input
        assert isinstance(p1, np.ndarray)
        assert isinstance(p2, np.ndarray)
        assert p1.shape == p2.shape
        assert len(p1.shape) == 2

        # 1) Cycle for each pair of endpoints
        ind = np.array([], dtype=int)
        label = np.array([], dtype=int)
        for line_number in range(len(p1)):

            # Point coordinates
            p1r = p1[line_number, 0]
            p1c = p1[line_number, 1]
            p2r = p2[line_number, 0]
            p2c = p2[line_number, 1]

            # 2) Boundary verification
            # 2.1) A- Both points are out of range
            if ((p1r < 1 or M < p1r) or (p1c < 1 or N < p1c)) and ((p2r < 1 or M < p2r) or (p2c < 1 or N < p2c)):
                raise RuntimeError(f"Both points in line segment {line_number} are out of range. "
                                   f"New coordinates are requested to fit the points in image boundaries.")

            # 3) Reference versors
            #      .....r..c.....
            eN = np.array([[-1,  0]]).T
            eE = np.array([[0,  1]]).T
            eS = np.array([[1,  0]]).T
            eW = np.array([[0, - 1]]).T

            # 4) B- One of the points is out of range
            if (p1r < 1 or M < p1r) or (p1c < 1 or N < p1c) or (p2r < 1 or M < p2r) or (p2c < 1 or N < p2c):
                # 4.1) Classify the inner and outer point
                if (p1r < 1 or M < p1r) or (p1c < 1 or N < p1c):
                    p_out = np.array([[p1r, p1c]]).T
                    p_in = np.array([[p2r, p2c]]).T
                elif (p2r < 1 or M < p2r) or (p2c < 1 or N < p2c):
                    p_out = np.array([[p2r, p2c]]).T
                    p_in = np.array([[p1r, p1c]]).T

                # 4.2) Vector defining line segment
                v = p_out - p_in
                aux = np.sort(np.abs(v), axis=0)
                aspect_ratio = aux[0, 0] / aux[1, 0]

                # 4.3) Vector orientation
                north = (v.T @ eN)[0, 0]
                west = (v.T @ eW)[0, 0]
                east = (v.T @ eE)[0, 0]
                south = (v.T @ eS)[0, 0]

                # 4.4) Increments
                deltaNS = 0
                if north > 0:
                    deltaNS = -1
                if south > 0:
                    deltaNS = 1

                deltaWE = 0
                if east > 0:
                    deltaWE = 1
                if west > 0:
                    deltaWE = -1

                # 4.5) Matrix subscripts occupied by the line segment
                if abs(v[0]) >= abs(v[1]):
                    alpha = [p_in[0, 0]]
                    beta = [p_in[1, 0]]
                    iter = 0
                    while 0 <= alpha[iter] <= (M-1) and 0 <= beta[iter] <= (N - 1):
                        alpha.append(alpha[iter] + deltaNS )              # alpha grows throughout the column direction.
                        beta.append(beta[iter] + aspect_ratio * deltaWE)  # beta grows throughout the row direction.
                        iter += 1
                    alpha = np.round(np.array(alpha[:-1])).astype(int)
                    beta = np.round(np.array(beta[:-1])).astype(int)
                    ind = np.append(ind, np.ravel_multi_index((alpha, beta), (N, M), order='F'))
                    label = np.append(label, line_number * np.ones(len(alpha), dtype=int))

                if abs(v[0]) < abs(v[1]):
                    alpha = [p_in[1, 0]]
                    beta = [p_in[0, 0]]
                    iter = 0
                    while 0 <= alpha[iter] <= (N - 1) and 0 <= beta[iter] <= (M - 1):
                        alpha.append(alpha[iter] + deltaWE )              # alpha grows throughout the row direction.
                        beta.append(beta[iter] + aspect_ratio * deltaNS)  # beta grows throughout the column direction.
                        iter += 1
                    alpha = np.round(np.array(alpha[:-1])).astype(int)
                    beta = np.round(np.array(beta[:-1])).astype(int)
                    ind = np.append(ind, np.ravel_multi_index((beta, alpha), (N, M), order='F'))
                    label = np.append(label, line_number * np.ones(len(alpha), dtype=int))
                del alpha, beta
                continue

            # 5) C- Both points are in range
            # 5.1) Classify the inner and outer point
            p_out = np.array([[p2r, p2c]]).T
            p_in = np.array([[p1r, p1c]]).T

            # 5.2) Vector defining line segment
            v = p_out - p_in
            aux = np.sort(np.abs(v), axis=0)
            aspect_ratio = aux[0, 0] / aux[1, 0]

            # 5.3) Vector orientation.
            north = (v.T @ eN)[0, 0]
            west = (v.T @ eW)[0, 0]
            east = (v.T @ eE)[0, 0]
            south = (v.T @ eS)[0, 0]

            # 5.4) Increments
            deltaNS = 0
            if north > 0:
                deltaNS = -1
            if south > 0:
                deltaNS = 1

            deltaWE = 0
            if east > 0:
                deltaWE = 1
            if west > 0:
                deltaWE = -1

            # 5.5) Matrix subscripts occupied by the line segment
            row_range = np.sort(np.array([p1r, p2r]))
            col_range = np.sort(np.array([p1c, p2c]))
            if abs(v[0]) >= abs(v[1]):
                alpha = [p_in[0, 0]]
                beta = [p_in[1, 0]]
                iter = 0
                while row_range[0] <= alpha[iter] <= row_range[1] and col_range[0] <= beta[iter] <= col_range[1]:
                    alpha.append(alpha[iter] + deltaNS)               # alpha grows throughout the column direction.
                    beta.append(beta[iter] + aspect_ratio * deltaWE)  # beta grows throughout the row direction.
                    iter += 1
                alpha = np.round(np.array(alpha[:-1])).astype(int)
                beta = np.round(np.array(beta[:-1])).astype(int)
                ind = np.append(ind, np.ravel_multi_index((alpha, beta), (N, M), order='F'))
                label = np.append(label, line_number * np.ones(len(alpha), dtype=int))
            if abs(v[0]) < abs(v[1]):
                alpha = [p_in[1, 0, 0]]
                beta = [p_in[0, 0]]
                iter = 0
                while col_range[0] <= alpha[iter] <= col_range[1] and row_range[0] <= beta[iter] <= row_range[1]:
                    alpha.append(alpha[iter] + deltaWE)  # alpha grows throughout the row direction.
                    beta.append(beta[iter] + aspect_ratio * deltaNS)  # beta grows throughout the column direction.
                    iter += 1
                alpha = np.round(np.array(alpha[:-1])).astype(int)
                beta = np.round(np.array(beta[:-1])).astype(int)
                ind = np.append(ind, np.ravel_multi_index((beta, alpha), (N, M), order='F'))
                label = np.append(label, line_number * np.ones(len(alpha), dtype=int))
            del alpha, beta
            continue

        return ind, label




