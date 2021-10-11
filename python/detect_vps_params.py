import numpy as np


class Params(object):
    def __init__(self, acceleration: bool = True, manhattan: bool = False):
        # 1) define default algorithm params
        self.__ppd = np.array([2., 2.])  # principal point = [W H]./params.ppd; values taken from YUD
        self.__REFINE_THRESHOLD = 2.  # (\theta)
        self.__VARIATION_THRESHOLD = 0.1  # (\zeta)
        self.__DUPLICATES_THRESHOLD = 0.0001  # (\delta)
        self.__VERTICAL_ANGLE_THRESHOLD = 50.  # (\omega)
        self.__INFINITY_THRESHOLD = 3.6  # (\lambda)
        self.__DEFAULT_SEGMENT_LENGTH_THRESHOLD = 1.71  # (\tau) final threshold will be sqrt(W+H)/tau
        self.__DIST_THRESHOLD = .14  # (\kappa)

        # 2) define algorithm "flexible" params
        self.__ORTHOGONALITY_THRESHOLD = 0.
        self.__MANHATTAN = False
        self.__ACCELERATION = False
        self.MANHATTAN = manhattan                       # this will force the re-calculation of ORTHOGONALITY_THRESHOLD
        self.ACCELERATION = acceleration

        # 3) define algorithm acceleration params
        self.__MAX_POINTS_ACCELERATION = 200  # use acceleration if number of points is larger than this
        self.__GMM_Ks = np.array([30, 30, 30])

        # 4) define parameters that change during runtime
        # self.__W = int(-1)
        # self.__H = int(-1)
        # self.__LENGTH_THRESHOLD = -np.inf
        # self.__img_in = ''
        # self.__folder_out = ''
        # self.__plot = False
        # self.__print = False
        # self.__FOCAL_RATIO = -np.inf

    @property
    def ppd(self):
        return self.__ppd

    @ppd.setter
    def ppd(self, val):
        assert isinstance(val, np.ndarray)
        assert val.dtype == float
        assert len(val) == 2
        self.__ppd = val

    @property
    def REFINE_THRESHOLD(self):
        return self.__REFINE_THRESHOLD

    @REFINE_THRESHOLD.setter
    def REFINE_THRESHOLD(self, val):
        assert isinstance(val, float)
        self.__REFINE_THRESHOLD = val

    @property
    def VARIATION_THRESHOLD(self):
        return self.__VARIATION_THRESHOLD

    @VARIATION_THRESHOLD.setter
    def VARIATION_THRESHOLD(self, val):
        assert isinstance(val, float)
        self.__VARIATION_THRESHOLD = val

    @property
    def DUPLICATES_THRESHOLD(self):
        return self.__DUPLICATES_THRESHOLD

    @DUPLICATES_THRESHOLD.setter
    def DUPLICATES_THRESHOLD(self, val):
        assert isinstance(val, float)
        self.__DUPLICATES_THRESHOLD = val

    @property
    def VERTICAL_ANGLE_THRESHOLD(self):
        return self.__VERTICAL_ANGLE_THRESHOLD

    @VERTICAL_ANGLE_THRESHOLD.setter
    def VERTICAL_ANGLE_THRESHOLD(self, val):
        assert isinstance(val, float)
        self.__VERTICAL_ANGLE_THRESHOLD = val

    @property
    def INFINITY_THRESHOLD(self):
        return self.__INFINITY_THRESHOLD

    @INFINITY_THRESHOLD.setter
    def INFINITY_THRESHOLD(self, val):
        assert isinstance(val, float)
        self.__INFINITY_THRESHOLD = val

    @property
    def DEFAULT_SEGMENT_LENGTH_THRESHOLD(self):
        return self.__DEFAULT_SEGMENT_LENGTH_THRESHOLD

    @DEFAULT_SEGMENT_LENGTH_THRESHOLD.setter
    def DEFAULT_SEGMENT_LENGTH_THRESHOLD(self, val):
        assert isinstance(val, float)
        self.__DEFAULT_SEGMENT_LENGTH_THRESHOLD = val

    @property
    def DIST_THRESHOLD(self):
        return self.__DIST_THRESHOLD

    @DIST_THRESHOLD.setter
    def DIST_THRESHOLD(self, val):
        assert isinstance(val, float)
        self.__DIST_THRESHOLD = val

    @property
    def MANHATTAN(self):
        return self.__MANHATTAN

    @MANHATTAN.setter
    def MANHATTAN(self, setting: bool):
        assert isinstance(setting, bool)
        self.__MANHATTAN = setting
        if self.__MANHATTAN:
            self.__ORTHOGONALITY_THRESHOLD = np.cos(87.5 * (np.pi / 180))  # (\gamma_S)
        else:
            self.__ORTHOGONALITY_THRESHOLD = np.cos(77.5 * (np.pi / 180))  # (\gamma_R)

    @property
    def ACCELERATION(self):
        return self.__ACCELERATION

    @ACCELERATION.setter
    def ACCELERATION(self, setting):
        assert isinstance(setting, bool)
        self.__ACCELERATION = setting

    @property
    def ORTHOGONALITY_THRESHOLD(self):
        return self.__ORTHOGONALITY_THRESHOLD

    @property
    def MAX_POINTS_ACCELERATION(self):
        return self.__MAX_POINTS_ACCELERATION

    @MAX_POINTS_ACCELERATION.setter
    def MAX_POINTS_ACCELERATION(self, val):
        assert isinstance(val, int)
        self.__MAX_POINTS_ACCELERATION = val

    @property
    def GMM_Ks(self):
        return self.__GMM_Ks

    @GMM_Ks.setter
    def GMM_Ks(self, val):
        assert isinstance(val, np.ndarray)
        assert val.dtype == int
        assert len(val) == 3
        self.__GMM_Ks = val

    # def GetConfig(self, as_string: bool = False) -> dict or str:
    #     properties = [key for key in dir(self) if (not ('__' in key) and not callable(getattr(self, key)))]
    #     assert len(properties) > 0  # TODO: get exact number of params
    #     if as_string:
    #         return '\n'.join([f'{prop} = {getattr(self, prop)}' for prop in properties])
    #     else:
    #         out_dict = {}
    #         for prop in properties:
    #             out_dict[prop] = getattr(self, prop)
    #         return out_dict

    def Get(self) -> dict:
        class_items = self.__class__.__dict__.items()
        return dict((k, getattr(self, k))
                    for k, v in class_items
                    if isinstance(v, property))

    def __str__(self) -> str:
        return '\n'.join([f'{key} = {val}' for key, val in self.Get().items()])

    def SetConfig(self, params_in: dict) -> None:
        assert isinstance(params_in, dict)
        properties = [key for key in dir(self) if (not ('__' in key) and not callable(getattr(self, key)))]
        for key in list(params_in.keys()):
            if key in properties:
                setattr(self, key, params_in[key])
            else:
                raise ValueError(f"`{key}` is not a known parameter")

    def properties(self):
        class_items = self.__class__.__dict__.items()
        return dict((k, getattr(self, k))
                    for k, v in class_items
                    if isinstance(v, property))


class RuntimeParams(Params):
    def __init__(self, acceleration: bool = True, manhattan: bool = False):
        # 1) call constructor of superclass
        super().__init__(acceleration, manhattan)

        # 2) define parameters that change during runtime
        self.__W = int(-1)
        # self.__H = int(-1)
        # self.__LENGTH_THRESHOLD = -np.inf
        # self.__img_in = ''
        # self.__folder_out = ''
        # self.__plot = False
        # self.__print = False
        # self.__FOCAL_RATIO = -np.inf

    def GetConfig(self) -> dict:
        return self.Get(self.__class__.__base__)

    @property
    def W(self):
        return self.__W

    @W.setter
    def W(self, val):
        assert isinstance(val, int)
        self.__W = val


if __name__ == '__main__':
    par = Params()
    rr = RuntimeParams()
    cfg = rr.GetConfig()
    print("OK")