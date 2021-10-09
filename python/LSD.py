import os
import cv2
import numpy as np
from cffi import FFI

# define the directory of current file
mydir = os.path.dirname(os.path.abspath(__file__))


# ############################################################
# Build function for the LSD (line segment detector) c library
# ############################################################
def __build_lsd():
    """
    This functions compiles the Line Segment Detector (LSD) algorithm from c source files using the `cffi` package.

    Important notes:
                    1) The function requires a compiler in order to run.
                       In linux, this means gcc.
                       In windows, this means that visual studio must be installed, with an MSVC version compatible with
                       the python version (2019 for python 3.7+). When installing visual studio, you must also make sure
                       that the following packages are installed:
                       * Visual C++ Build tools core features
                       * MSVC toolset C++ (for current version)
                       * Visual C++ Redistributable Update (for current version)
                       * Windows 10 SDK for Desktop C++ (latest version)
                    2) DO NOT (!!!) try to run this function using PyCharm. Use a regular console without a debugger or
                       anything attached to it. The reason is that these consoles often modify or neglect to use some
                       system variables that are required for the compilation through cffi to work.
                       AFTER the library has been compiled, it can be imported using pycharm.

    When the function is finished, there should be two new files in the folder of this file:
    1) _lsd.c - a new c file of the sources, compiled in the cffi format.
    2) _lsd.xxxx.pyd - the pythonic shared object of the algorithm library, that is loaded during runtime
                       (xxxx refers to the python and operation system versions).
    :return: None
    """
    # get absolute paths of lsd c source files

    path_to_header = mydir + r'/../VPdetection-CVPR14/mex_files/lsd.h'
    path_to_source = mydir + r'/../VPdetection-CVPR14/mex_files/lsd.c'

    # read header file
    with open(path_to_header, 'r') as f:
        header = f.read()

    # remove definitions that cannot be handled by cffi
    header = header.replace('#ifndef LSD_HEADER', '')
    header = header.replace('#define LSD_HEADER', '')
    header = header.replace('#endif', '')

    # read source file
    with open(path_to_source, 'r') as f:
        source = f.read()

    # remove all MATLAB related stuff from source
    source = source.replace('#include "lsd.h"', '')
    source = source.replace('#include <mex.h>', '')
    mex_function_loc = source.find('void mexFunction')
    source = source[:mex_function_loc]

    # use cffi to compile the code
    ffibuilder = FFI()
    ffibuilder.cdef(header)

    ffibuilder.set_source("_lsd", source)

    try:
        ffibuilder.compile(verbose=True)
    except:
        raise RuntimeError("Could not compile the LSD library. Verify required compilers are installed correctly.")


# ############################################################
# Library import
# ############################################################
# try to import the compiled LSD library. If it does not succeed, try to compile it and then try to import it again.
try:
    from _lsd import ffi, lib
except:
    __build_lsd()
    try:
        from _lsd import ffi, lib
    except:
        raise RuntimeError("Could not import the LSD library even though it compiled without errors.")


# ############################################################
# Wrapper class for the LSD algorithm
# ############################################################
class LSDWrapper:
    """
    This class wraps the LSD algorithm library compiled with cffi in a user-friendly way.
    Instead of converting/casting back and from cffi types, the user can simply use the numpy array format.

    The class contains a single usable function called `run_lsd` which gets a single image in a numpy array format.
    The reason the function was wrapped in a class is that the buffer returned from the lsd library function will be
    deleted by the python garbage collector one the function returns, and in order to save time and not copy the buffer,
    the buffer will be saves as a class member. e.g it will continue to live as long as the instance of the class lives.
    """
    def __init__(self):
        self._num_of_lines = ffi.new("int[1]")
        self._elemets_per_line = int(7)
        self._buffer = None                 # placeholder for the output data of the c function
        self._lines = None                  # placeholder for the returned numpy array

    def run_lsd(self, image_in: np.ndarray, cut_results: bool = True):
        """
        Runs the Line Segment Detector algorithm on a single image.
        :param image_in: a numpy array of an image (at least two dimensions). Can be an RGB/BGR image.
        :param cut_results: If true (default), the output will contain only the data relevant for the vanishing point
                            detection algorithm. If false, the full results of the algorithm will be returned.
        :return: an nx4 (cut_results=True) or nx7 (cut_results=False) numpy array, where n is the number of detected
                 line segments in the image.
        """
        # verify input is correct
        assert (isinstance(image_in, np.ndarray))
        assert (len(image_in.shape) == 2 or len(image_in.shape) == 3)

        # convert RGB image to grayscale if needed
        if len(image_in.shape) == 3:
            assert (image_in.shape[2] == 3)
            image_gray = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image_in

        # convert uint8 image to double if needed
        if image_gray.dtype != np.float64:
            image_double = image_gray.astype(np.float64)
        else:
            image_double = image_gray

        # call the algorithm
        self._buffer = lib.lsd(ffi.cast("int *", ffi.addressof(self._num_of_lines)),
                               ffi.cast("double *", image_double.ctypes.data),
                               image_double.shape[1],
                               image_double.shape[0])

        # convert the returned buffer to numpy array (and reshape the resulting array into an nx7 array)
        self._lines = np.frombuffer(ffi.buffer(self._buffer,
                                               self._num_of_lines[0] * self._elemets_per_line * ffi.sizeof('double')),
                                    np.float64).reshape((self._num_of_lines[0], self._elemets_per_line))
        if cut_results:
            return self._lines[:, :4]
        else:
            return self._lines
