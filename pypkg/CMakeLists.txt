cmake_minimum_required(VERSION 3.4...3.18)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

project(_gl0learn)

FIND_PACKAGE(BLAS REQUIRED)
FIND_PACKAGE(LAPACK REQUIRED)

FIND_LIBRARY(ARMADILLO armadillo HINTS /usr/local/lib/ REQUIRED)

ADD_SUBDIRECTORY(external/pybind11)
ADD_SUBDIRECTORY(external/carma)

pybind11_add_module(_gl0learn MODULE
    gl0learn/src/pygl0learn.cpp
    gl0learn/src/pyoracle.cpp
    gl0learn/src/pyfitmodel.cpp
    gl0learn/src/pyfit.cpp
)


TARGET_LINK_LIBRARIES(_gl0learn PRIVATE carma::carma BLAS::BLAS LAPACK::LAPACK)

set_target_properties(_gl0learn PROPERTIES OUTPUT_NAME gl0learn/_gl0learn)

target_include_directories(_gl0learn PRIVATE gl0learn/src/include)
target_include_directories(_gl0learn PRIVATE gl0learn/src)
