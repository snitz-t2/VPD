import numpy as np
from copy import deepcopy

# define the smallest possible floating point number, to be used as an `epsilon` in following functions
realmin = np.finfo(float).tiny


def run_mixtures(points: np.ndarray, Ks: np.ndarray) -> np.ndarray:
    """
    Runs Figueiredo et al. GMM algorithm with different parameters (number of Gaussians).
    The endpoints of the ellipses found are saved as candidates for the segment alignment detector.
    :param points: An Nx2 numpy array, where `N` is the number of segment endpoints.
    :param Ks: Number of Gaussians to try (example: [20 40 60]).
    :return: all_bestpairs - An Mx2 numpy array, where `M` is the number of aligning segment endpoints detected by
                             the gaussian mixtures model.
    """
    # TODO
    unique_points = np.unique(np.round(points).astype(float), axis=0)

    all_bestpairs = np.empty((0, 2), dtype=np.float64)
    for K in Ks:
        bestK, bestpp, bestmu, bestcov, dl, countf, bestpairs = mixtures4(unique_points.T,
                                                                          kmin=max(2, K-7), kmax=K, regularize=0.,
                                                                          th=1e-4, covoption=0)
        all_bestpairs = np.append(all_bestpairs, bestpairs, axis=0)

    # TODO: re-accomodate results

    return all_bestpairs


def mixtures4(y: np.ndarray,
              kmin: int,
              kmax: int,
              regularize: float,
              th: float,
              covoption: int,
              randindex: np.ndarray = None) -> tuple:
    """
    Performs a Gaussian Mixtures Model (GMM) clustering algorithm on the provided datapoints.
    Parameters of each distributions are retured, as well as the endpoints of each distribution ellipse.

    :param y: the data; for n observations in d dimensions, y must have d lines and n columns.
    :param kmin: the minimum number of components to be tested
    :param kmax: the initial (maximum) number of mixture components
    :param regularize: a regularizing factor for covariance matrices. in very small samples,
                       it may be necessary to add some small quantity to the diagonal of the covariances.
    :param th: stopping threshold
    :param covoption: controls the covarince matrix. there are four options:
                        - covoption = 0 means free covariances
                        - covoption = 1 means diagonal covariances
                        - covoption = 2 means a common covariance for all components
                        - covoption = 3 means a common diagonal covarince for all components
    :param randindex: DO NOT USE, it is only for testing. Should be None otherwise!!!
    :return: A length 7 tuple with the following results in the following order:
             - bestK: int.
                      The selected number of components.
             - bestpp: Numpy array with size of 1xbestK (float).
                       The obtained vector of mixture probabilities.
             - bestmu: Numpy array with size of dxbestK (where d is the number of dimensions of the data).
                       Contains the estimates of the means of the components.
             - bestcov: Numpy array with size of dxdxbestK.
                        Contains the estimates of the covariances of the components.
             - dl: dict.
                   Contains the successive values of the cost function.
                   Keys of this dict are the iteration numbers where the results were obtained.
             - countf: int.
                       The total number of iterations performed.
             - bestpairs: Numpy array with size of bestKx4.
                          Each line represents the two endpoints of the ellipse of a mixture component ([x1 y1 x2 y2]).
                          This is the only output needed for the vanishing point detection algorithm.
    """
    # 0) handle input:
    assert (isinstance(y, np.ndarray))
    assert (isinstance(kmin, int) and kmin > 0)
    assert (isinstance(kmax, int) and kmax > 0 and kmax > kmin)
    assert (isinstance(regularize, float))
    assert (isinstance(th, float))
    assert (isinstance(covoption, int))

    # 1) define function parameters
    dl = []                                 # list to store the consecutive values of the cost function
    dimensions, nPoints = y.shape

    # 2) switch covoptions
    if covoption == 0:
        npars = (dimensions + dimensions * (dimensions + 1) / 2)    # this is for free covariance matrices
    elif covoption == 1:
        npars = 2 * dimensions                                      # this is for diagonal covariance matrices
    elif covoption == 2:
        npars = dimensions              # this is for a common covariance matrix independently of its structure
    elif covoption == 3:
        npars = dimensions
    else:
        npars = (dimensions + dimensions * (dimensions + 1) / 2)    # the default is to assume free covariances
    nparsover2 = npars / 2

    # 3) kmax is the initial number of mixture components
    k = kmax

    # 4) indic will contain the assignments of each data point to the mixture components, as result of the E-step
    indic = np.zeros((k, nPoints))
    semi_indic = np.zeros((k, nPoints))

    # 5) Initialization: we will initialize the means of the k components with k randomly chosen data points.
    #                    That generates random permutations of the integers from 1 to n.
    if randindex is None:
        randindex = np.random.permutation(nPoints)
    randindex = randindex[:k]
    estmu = y[:, randindex]

    # 6) the initial estimates of the mixing probabilities are set to 1/k
    estpp = (1 / k) * np.ones((1, k))

    # 7) here we compute the global covariance of the data
    globcov = np.cov(y)

    # 8) calculate the covariances for k permutations
    #    they are initialized to diagonal matrices proportional to 1/10 of the mean variance along all the axes.
    #    Of course, this can be changed.
    estcov = np.zeros((*globcov.shape, k), dtype=float)
    for ii in range(k):
        estcov[:, :, ii] = np.diag(np.ones(dimensions) * max(np.diag(globcov / 10)))

    # 9) having the initial means, covariances, and probabilities, we can initialize the indicator functions
    #    following the standard EM equation. Notice that these are unnormalized values.
    for ii in range(k):
        semi_indic[ii, :] = my_multinorm(y, estmu[:, ii][:, np.newaxis], estcov[:, :, ii], nPoints)
        indic[ii, :] = semi_indic[ii, :] * estpp[0, ii]

    # 10) we can use the indic variables (unnormalized) to compute the loglikelihood and store it
    #     for later plotting its evolution. we also compute and store the number of components.
    countf = 0
    loglike = {countf: sum(np.log(np.sum(realmin + indic, axis=0)))}
    dlength = -loglike[countf] + nparsover2 * np.sum(np.log(estpp)) + (nparsover2 + 0.5) * k * np.log(nPoints)
    dl = {countf: dlength}
    kappas = {countf: k}

    # 11) the transitions vectors will store the iteration number at which components are killed.
    #     transitions1 stores the iterations at which components are killed by the M-step,
    #     while transitions2 stores the iterations at which we force components to zero.
    transitions1 = []
    transitions2 = []

    # 12) minimum description length seen so far, and corresponding parameter estimates
    mindl = dl[countf]
    bestpp = deepcopy(estpp)
    bestmu = deepcopy(estmu)
    bestcov = deepcopy(estcov)
    bestk = deepcopy(k)

    # 13) start looping over gaussians
    k_cont = True      # auxiliary variable for the outer loop
    while k_cont:      # the outer loop will take us down from kmax to kmin components
        cont = True    # auxiliary variable of the inner loop
        while cont:    # this inner loop is the component-wise EM algo with the modified M-step that can kill components
            # 13.1) we begin at component 1 and can only go to the last component, k.
            #       Since k may change during the process, we can not use a for loop.
            comp = 0
            while comp <= (k - 1):
                # 13.1.1) we start with the M step. first, we compute a normalized indicator function
                indic = np.tile(estpp.T, [1, 418]) * semi_indic

                # 13.1.2) normalize indicator array
                normindic = indic / (realmin + np.tile(np.sum(indic, axis=0)[np.newaxis, :], [k, 1]))

                # 13.1.3) now we perform the standard M-step for mean and covariance
                normalize = 1 / np.sum(normindic[comp, :])
                aux = np.tile(normindic[comp, :], [dimensions, 1]) * y
                estmu[:, comp] = normalize * np.sum(aux, axis=1)
                estcov[:, :, comp] = normalize * (aux @ y.T) - \
                                     estmu[:, comp][:, np.newaxis] @ estmu[:, comp][:, np.newaxis].T + \
                                     regularize * np.eye(dimensions)

                # 13.1.4) this is the special part of the M step that is able to kill components
                #         it is done by zeroing the value for current component in `estpp`
                estpp[:, comp] = max(np.sum(normindic[comp, :]) - nparsover2, 0) / nPoints
                estpp /= np.sum(estpp)

                # 13.1.5) this is an auxiliary variable that will be used the
                #         signal the killing of the current component being updated
                killed = False

                # 13.1.6) we now have to do some book-keeping if the current component was killed.
                #         that is, we have to rearrange the vectors and matrices that store the parameter estimates.
                if estpp[0, comp] == 0 or np.isnan(estpp[:, comp]):
                    killed = True

                    # we also register that at the current iteration a component was killed
                    transitions1.append(countf)

                    if comp == 0:
                        estmu = estmu[:, 1:]
                        estcov = estcov[:, :, 1:]
                        estpp = estpp[:, 1:]
                        semi_indic = semi_indic[1:, :]
                    else:
                        if comp == k-1:
                            estmu = estmu[:, :-1]
                            estcov = estcov[:, :, :-1]
                            estpp = estpp[:, :-1]
                            semi_indic = semi_indic[:-1, :]
                        else:
                            estmu = np.hstack([estmu[:, :comp], estmu[:, (comp+1):]])
                            newcov = np.zeros((dimensions, dimensions, k-1), dtype=float)
                            for kk in range(comp):
                                newcov[:, :, kk] = estcov[:, :, kk]
                            for kk in range(comp+1, k):
                                newcov[:, :, kk-1] = estcov[:, :, kk]
                            estcov = newcov
                            estpp = np.hstack([estpp[:, :comp], estpp[:, (comp + 1):]])
                            semi_indic = np.vstack([semi_indic[:comp, :], semi_indic[(comp + 1):, :]])

                    # 13.1.7) since we've just killed a component, k must decrease
                    k -= 1

                if not killed:
                    # 13.1.8) if the component was not killed, we update the corresponding indicator variables...
                    semi_indic[comp, :] = my_multinorm(y, estmu[:, comp][:, np.newaxis], estcov[:, :, comp], nPoints)

                    # 13.1.9) ...and move on to the next component
                    comp += 1
                    # Note: if killed==1, it means the in the position "comp",
                    #       we now have a component that was not yet visited in this sweep,
                    #       and so all we have to do is go back to the M step without increasing "comp".

            # ----- this is the end of the innermost "while comp <= k" loop, which cycles through the components -------

            # 13.2) increment the iterations counter
            countf += 1

            # 13.3) clear and re-calculate indic (if k has been changed, the following matrices will be different)
            indic = np.zeros((k, nPoints))
            semi_indic = np.zeros((k, nPoints))
            for ii in range(k):
                semi_indic[ii, :] = my_multinorm(y, estmu[:, ii][:, np.newaxis], estcov[:, :, ii], nPoints)
                indic[ii, :] = semi_indic[ii, :] * estpp[0, ii]

            # 13.4) compute the loglikelihhod function and the description length
            if k != 1:
                # if the number of surviving components is not just one, we compute the loglikelihood from
                # the unnormalized assignment variables
                loglike[countf] = sum(np.log(np.sum(realmin + indic, axis=0)))
            else:
                # if it is just one component, it is even simpler
                # TODO: verify correctness. the expression above may be true for both cases
                loglike[countf] = sum(np.log(realmin + indic))

            # 13.5) compute and store the description length and the current number of components
            dlength = -loglike[countf] + nparsover2 * np.sum(np.log(estpp)) + (nparsover2 + 0.5) * k * np.log(nPoints)
            dl[countf] = dlength
            kappas[countf] = k

            # 13.6) compute the change in loglikelihood to check if we should stop
            #       if the relative change in loglikelihood is below the threshold, we stop CEM2
            deltlike = loglike[countf] - loglike[countf - 1]
            if np.abs(deltlike / loglike[countf - 1]) < th:
                cont = False

        # ---------------- this end is of the inner loop: "while(cont)" ------------------------------------------------

        # 14) now check if the latest description length is the best.
        #     if it is, we store its value and the corresponding estimates
        if dl[countf] < mindl:
            bestpp = deepcopy(estpp)
            bestmu = deepcopy(estmu)
            bestcov = deepcopy(estcov)
            bestk = deepcopy(k)
            mindl = dl[countf]

        # 15) at this point, we may try smaller mixtures by killing the component with the smallest mixing probability
        #     and then restarting CEM2, as long as k is not yet at kmin.
        if k > kmin:
            indminp = np.argmin(estpp)

            # 15.1) what follows is the book-keeping associated with removing one component
            if indminp == 0:
                estmu = estmu[:, 1:]
                estcov = estcov[:, :, 1:]
                estpp = estpp[:, 1:]
            else:
                if indminp == k:
                    estmu = estmu[:, :-1]
                    estcov = estcov[:, :, :-1]
                    estpp = estpp[:, :-1]
                else:
                    estmu = np.hstack([estmu[:, :indminp], estmu[:, (indminp + 1):]])
                    newcov = np.zeros((dimensions, dimensions, k - 1), dtype=float)
                    for kk in range(indminp):
                        newcov[:, :, kk] = estcov[:, :, kk]
                    for kk in range(indminp + 1, k):
                        newcov[:, :, kk - 1] = estcov[:, :, kk]
                    estcov = newcov
                    estpp = np.hstack([estpp[:, :indminp], estpp[:, (indminp + 1):]])
            k -= 1

            # 16) we renormalize the mixing probabilities after killing the component...
            estpp /= np.sum(estpp)

            # 17) ...and register the fact that we have forced one component to zero
            transitions2.append(countf)

            # 18) increment the iterations counter...
            countf += 1

            # 19) ...and compute the loglikelihhod function and the description length
            indic = np.zeros((k, nPoints))
            semi_indic = np.zeros((k, nPoints))
            for ii in range(k):
                semi_indic[ii, :] = my_multinorm(y, estmu[:, ii][:, np.newaxis], estcov[:, :, ii], nPoints)
                indic[ii, :] = semi_indic[ii, :] * estpp[0, ii]
            if k != 1:
                loglike[countf] = sum(np.log(np.sum(realmin + indic, axis=0)))
            else:
                loglike[countf] = sum(np.log(realmin + indic))
            dl[countf] =  -loglike[countf] + nparsover2 * np.sum(np.log(estpp)) + (nparsover2 + 0.5) * k * np.log(nPoints)
            kappas[countf] = k

        # this else corresponds to "if k > kmin".
        else:
            k_cont = False                      # of course, if k is not larger than kmin, we must stop.

    # ---------------- this end is of the outer loop: "while(k_cont)" --------------------------------------------------

    # 20) get elipses endpoints
    bestpairs = np.zeros((len(bestpp.T), 4))
    for comp in range(len(bestpp.T)):
        bestpair = get_ellipse_endpoints(bestmu[:, comp], bestcov[:, :, comp], 2)
        bestpairs[comp, :] = bestpair

    # 21) finish the function - return parameters
    return bestk, bestpp, bestmu, bestcov, dl, countf, bestpairs


# #################################################################################################################### #
# Help Functions
# #################################################################################################################### #
def my_multinorm(x: np.ndarray, m: np.ndarray ,covar: np.ndarray, npoints: int):
    """
    Evaluates (samples) a multidimensional Gaussian of mean m and covariance matrix covar at the array of points x.
    E.g for a (two dimensional) gaussian distribution with mean `m` and covariance `covar`, this function will return
    the values of the distribution at sample points `x`.
    :param x: A (2)x(npoints) numpy array (float). Each 2x1 vector within this array represents a location of a point.
    :param m: A 2x1 numpy array which represents the location of the mean.
    :param covar: A 2x2 numpy array that represents the covariance matrix of the distribution.
    :param npoints: The number of points to evaluate / ssample.
    :return: A (1)x(npoints) numpy array, where each value in the array is the sampled value of the distribution at the
             location in x with the corresponding index.
    """
    X = covar + realmin * np.eye(2)
    dd = np.linalg.det(X)
    inv = np.linalg.inv(X)
    ff = ((2 * np.pi) ** (-1)) * (dd ** (-0.5))
    centered = (x - m * np.ones(npoints))
    return ff * np.exp(-0.5 * np.sum(centered * (inv @ centered), axis=0))


def get_ellipse_endpoints(m: np.ndarray, cov: np.ndarray, level: float) -> np.ndarray:
    """
    Returns the two endpoints of  a bivariate gaussian density with mean "m" and covariance matrix "cov".
    The level is controlled by "level".
    The parameters `a` and `b` of the ellipse are obtained using SVD.
    The `level` parameter is used to control the width of the ellipse in relation to the STD of the distribution.
    :param m: Numpy array of length 2 (float). The mean of the bivariate gaussian.
    :param cov: A 2x2 numpy array (float). The covariance matrix of the distribution.
    :param level: float. This number multiplies the STD of the distribution in each principal axis of the ellipse
                         in order to get the `a` anf `b` parameters.
    :return: pair: Numpy array of length 4 (float).
                   This array represents two points in the following structure: [x1, y1, x2, y2].
                   The returned pair is the two endpoints of the ellipse.
    """
    [U, S, V] = np.linalg.svd(cov)
    a = np.sqrt(S[0] * level * level)
    b = np.sqrt(S[1] * level * level)
    theta = np.array([[0, np.pi]])
    xx = a * np.cos(theta)
    yy = b * np.sin(theta)
    cord = np.vstack([xx, yy])
    cord = U @ cord
    pair = np.array([cord[0, 0] + m[0], cord[1, 0] + m[1], cord[0, 1] + m[0], cord[1, 1] + m[1]])
    return pair
