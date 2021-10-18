import numpy as np


def __set_val_by_type(self, attr_name, val):
    val_type = type(getattr(self, attr_name))
    assert (isinstance(val, val_type))
    setattr(self, attr_name, val)


def _create_property(attr_name: str):
    assert isinstance(attr_name, str)
    return property(fget=lambda self: getattr(self, attr_name),
                    fset=lambda self, val: __set_val_by_type(self, attr_name, val))


class Params(object):
    def Get(self) -> dict:
        class_items = self.__class__.__dict__.items()
        return dict((k, getattr(self, k))
                    for k, v in class_items
                    if isinstance(v, property))

    def Set(self, input_params: dict) -> bool:
        assert isinstance(input_params, dict)
        properties_list = list(self.Get().keys())
        for key, val in input_params:
            if key in properties_list:
                setattr(self, key, val)
            else:
                raise ValueError(f"`{key}` is not a known parameter")

    def __str__(self) -> str:
        return '\n'.join([f'{key} = {val}' for key, val in self.Get().items()])


class DefaultParams(Params):
    def __init__(self, acceleration: bool = True, manhattan: bool = False):
        # 1) define default algorithm params
        self._ppd = np.array([2., 2.])  # principal point = [W H]./params.ppd; values taken from YUD
        self._REFINE_THRESHOLD = 2.  # (\theta)
        self._VARIATION_THRESHOLD = 0.1  # (\zeta)
        self._DUPLICATES_THRESHOLD = 0.0001  # (\delta)
        self._VERTICAL_ANGLE_THRESHOLD = 50.  # (\omega)
        self._INFINITY_THRESHOLD = 3.6  # (\lambda)
        self._SEGMENT_LENGTH_THRESHOLD_COEFF = 1.71  # (\tau) final threshold will be sqrt(W+H)/tau
        self._DIST_THRESHOLD = .14  # (\kappa)

        # 2) define algorithm "flexible" params
        self._ORTHOGONALITY_THRESHOLD = 0.
        self._MANHATTAN = False
        self._ACCELERATION = False
        self.MANHATTAN = manhattan                       # this will force the re-calculation of ORTHOGONALITY_THRESHOLD
        self.ACCELERATION = acceleration

        # 3) define algorithm acceleration params
        self._MAX_POINTS_ACCELERATION = int(200)        # use acceleration if number of points is larger than this
        self._GMM_Ks = np.array([30, 30, 30])

    # declare properties for regular params, for which the only operation to do in the setter is check type correctness
    REFINE_THRESHOLD = _create_property('_REFINE_THRESHOLD')
    VARIATION_THRESHOLD = _create_property('_VARIATION_THRESHOLD')
    DUPLICATES_THRESHOLD = _create_property('_DUPLICATES_THRESHOLD')
    VERTICAL_ANGLE_THRESHOLD = _create_property('_VERTICAL_ANGLE_THRESHOLD')
    INFINITY_THRESHOLD = _create_property('_INFINITY_THRESHOLD')
    SEGMENT_LENGTH_THRESHOLD_COEFF = _create_property('_SEGMENT_LENGTH_THRESHOLD_COEFF')
    DIST_THRESHOLD = _create_property('_DIST_THRESHOLD')
    ACCELERATION = _create_property('_ACCELERATION')
    MAX_POINTS_ACCELERATION = _create_property('_MAX_POINTS_ACCELERATION')

    # declare properties for special params, e.g of complex type or shape,
    # or that changing them automatically induces a change in other parameters
    @property
    def ppd(self):
        return self._ppd

    @ppd.setter
    def ppd(self, val):
        assert isinstance(val, np.ndarray)
        assert val.dtype == float
        assert len(val) == 2
        self._ppd = val

    @property
    def MANHATTAN(self):
        return self._MANHATTAN

    @MANHATTAN.setter
    def MANHATTAN(self, setting: bool):
        assert isinstance(setting, bool)
        self._MANHATTAN = setting
        if self._MANHATTAN:
            self._ORTHOGONALITY_THRESHOLD = np.cos(87.5 * (np.pi / 180))  # (\gamma_S)
        else:
            self._ORTHOGONALITY_THRESHOLD = np.cos(77.5 * (np.pi / 180))  # (\gamma_R)

    @property
    def ORTHOGONALITY_THRESHOLD(self):
        return self._ORTHOGONALITY_THRESHOLD

    @property
    def GMM_Ks(self):
        return self._GMM_Ks

    @GMM_Ks.setter
    def GMM_Ks(self, val):
        assert isinstance(val, np.ndarray)
        assert val.dtype == int
        assert len(val) == 3
        self._GMM_Ks = val


class RuntimeParams(Params):
    def __init__(self):

        # define parameters that change during runtime
        self._W = int(-1)
        self._H = int(-1)
        self._SEGMENT_LENGTH_THRESHOLD = -np.inf
        self._img_in = ''
        self._folder_out = ''
        self._plot_graphs = False
        self._print_results = False
        self._FOCAL_RATIO = -np.inf
        self._ANG = -np.inf
        self._endpoints = False

    # declare properties for regular params, for which the only operation to do in the setter is check type correctness
    W = _create_property('_W')
    H = _create_property('_H')
    FOCAL_RATIO = _create_property('_FOCAL_RATIO')
    SEGMENT_LENGTH_THRESHOLD = _create_property('_SEGMENT_LENGTH_THRESHOLD')
    img_in = _create_property('_img_in')
    folder_out = _create_property('_folder_out')
    plot_graphs = _create_property('_plot_graphs')
    print_results = _create_property('_print_results')
    ANG = _create_property('_ANG')
    endpoints = _create_property('_endpoints')
