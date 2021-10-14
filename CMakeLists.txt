cmake_minimum_required(VERSION 3.16.4 FATAL_ERROR)

# define project name
project(VPD_libs)

# find the matlab engine installation
find_package(Matlab)

# add the mex project for the LSD lib
matlab_add_mex(NAME lsd SRC ${CMAKE_CURRENT_LIST_DIR}/VPdetection-CVPR14/mex_files/lsd.c ${CMAKE_CURRENT_LIST_DIR}/VPdetection-CVPR14/mex_files/lsd.h)

# glob the source files for alignment libraries
file(GLOB_RECURSE INCLUDE_FILES ${CMAKE_CURRENT_LIST_DIR}/VPdetection-CVPR14/mex_files/lib/*.h)
file(GLOB_RECURSE SOURCE_FILES ${CMAKE_CURRENT_LIST_DIR}/VPdetection-CVPR14/mex_files/lib/*.c)

# add the mex projects for both alignment libraries
matlab_add_mex(NAME alignments_slow SRC ${INCLUDE_FILES} ${SOURCE_FILES} ${CMAKE_CURRENT_LIST_DIR}/VPdetection-CVPR14/mex_files/alignments_slow.cpp)
matlab_add_mex(NAME alignments_fast SRC ${INCLUDE_FILES} ${SOURCE_FILES} ${CMAKE_CURRENT_LIST_DIR}/VPdetection-CVPR14/mex_files/alignments_fast.cpp)

# add the pybind11 projects for both alignment libraries
add_subdirectory(pybind11)
pybind11_add_module(pyalignments_slow ${INCLUDE_FILES} ${SOURCE_FILES} ${CMAKE_CURRENT_LIST_DIR}/VPdetection-CVPR14/mex_files/alignments_slow.cpp)
target_compile_definitions(pyalignments_slow PUBLIC pybind11_project)
pybind11_add_module(pyalignments_fast ${INCLUDE_FILES} ${SOURCE_FILES} ${CMAKE_CURRENT_LIST_DIR}/VPdetection-CVPR14/mex_files/alignments_fast.cpp)
target_compile_definitions(pyalignments_fast PUBLIC pybind11_project)

# define installation folder
install(TARGETS lsd alignments_slow alignments_fast DESTINATION ${CMAKE_CURRENT_LIST_DIR}/VPdetection-CVPR14/mex_files)
install(TARGETS pyalignments_slow pyalignments_fast DESTINATION ${CMAKE_CURRENT_LIST_DIR}/python)