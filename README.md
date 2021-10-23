# Vanishing Point Detection

This code implements the algorithm [Vanishing Point Detection in Urban Scenes Using Point Alignments](https://www.ipol.im/pub/art/2017/148/)
in python, based on the original c / matlab [implementation](https://github.com/jlezama/VPdetection-CVPR14.git), which 
is used as a submodule.  

## Compilation
The core of the algorithm is written in c, so in order to work cross-platform, a cmake-based build system must be used.
In addition, a suitable python virtual environments must be set in order to compile one of the libraries 
(and run the algorithm in general).

#### Instructions
1) Make sure your system has a c/c++ compiler like gcc or MSVC installed, in a way that cmake can detect it.
IMPORTANT: the compiler MUST support the c++11 standard.
2) Make sure a python version is installed and is accessible through the system's PATH. cmake needs to detect this too.
   The preferable version of python is 3.7 and above, but the algorithm MAY work on previous versions.
3) Install the required python packages (installing to a virtual environment is preferable).
   The packages can be installed using the [python_requirements](python_requirements.txt) file using the command:  
   `pip install -r python_requirements.txt`
4) Configure cmake on the project folder. DO NOT change the installation paths, as compiled libraries for both Python and 
MATLAB need to be installed to specific folders.
5) Generate and compile the solution / makefiles.
6) Install the compiled libraries using the `install` solution in VS or `make install` in linux.
7) If you used a virtual environment, activate it and `cd` to the [tests](python/tests) folder.  
   DO NOT use pycharm or any other IDE at this stage, just the command line.
8) First run `pytest test_LSD.py`. This has to be run before all other tests as it should compile the python wrapper to 
the Line Segment Detector c library, which is required to run the algorithm. Again, DO NOT use pycharm, as the cffi 
compiler may not work well when used through anything other than a command line.
9) Run the `pytest` command in the folder. It should run all tests, and they should all pass.

## Usage
For an example of usage, please refer to the function `test__detect_vps` 
in [test_VanishingPointDetector.py](python/tests/test_VanishingPointDetector.py).

