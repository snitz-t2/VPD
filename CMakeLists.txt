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

# define installation folder
install(TARGETS lsd alignments_slow alignments_fast DESTINATION ${CMAKE_CURRENT_LIST_DIR}/VPdetection-CVPR14/mex_files)
