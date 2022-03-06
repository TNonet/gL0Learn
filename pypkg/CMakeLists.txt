cmake_minimum_required(VERSION 3.4...3.18)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

project(gl0learn_core VERSION 0.0.2)

FIND_PACKAGE(BLAS REQUIRED)
FIND_PACKAGE(LAPACK REQUIRED)

FIND_LIBRARY(ARMADILLO armadillo HINTS /usr/local/lib/ REQUIRED)

ADD_SUBDIRECTORY(external/pybind11)
ADD_SUBDIRECTORY(external/carma)

pybind11_add_module(gl0learn_core MODULE
    src/gl0learn/src/pygl0learn.cpp
    src/gl0learn/src/pyoracle.cpp
    src/gl0learn/src/pyfitmodel.cpp
    src/gl0learn/src/pyfit.cpp
)

target_include_directories(gl0learn_core PRIVATE src/gl0learn/src/include)
target_include_directories(gl0learn_core PRIVATE src/gl0learn/src)

TARGET_LINK_LIBRARIES(gl0learn_core PRIVATE carma::carma BLAS::BLAS LAPACK::LAPACK)

target_compile_definitions(gl0learn_core PRIVATE VERSION_INFO=${PROJECT_VERSION})

set_target_properties(gl0learn_core PROPERTIES OUTPUT_NAME gl0learn/gl0learn_core)

# install(TARGETS gl0learn_core LIBRARY DESTINATION .)