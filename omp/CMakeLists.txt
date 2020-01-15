# An Example Script for CUDA with CMake

# CUDA support was introduced in version 3.8 and 
# will not work if there is a version mismatch
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

# Set up project with C  language support
project(Project LANGUAGES C )

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_executable(jacobian jacobian.c f_eval2.h)
target_link_libraries(jacobian -lm)
