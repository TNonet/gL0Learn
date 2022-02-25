cmake_minimum_required(VERSION 3.4...3.18)
project(_gl0learn)

add_subdirectory(external/pybind11)
add_subdirectory(external/carma)

pybind11_add_module(_gl0learn SHARED
    gl0learn/src/pygl0learn.cpp
    gl0learn/src/pyoracle.cpp
)

target_include_directories(_gl0learn PRIVATE gl0learn/src/include)
target_include_directories(_gl0learn PRIVATE gl0learn/src)

TARGET_LINK_LIBRARIES(_gl0learn PRIVATE carma::carma)

TARGET_INCLUDE_DIRECTORIES(_gl0learn
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

TARGET_COMPILE_OPTIONS(_gl0learn
    PUBLIC
        "$<$<CONFIG:RELEASE>:${PROJECT_RELEASE_FLAGS}>"
)

TARGET_COMPILE_DEFINITIONS(_gl0learn
    PUBLIC
        "$<$<CONFIG:RELEASE>:${PROJECT_RELEASE_DEFINITIONS}>"
)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(_gl0learn
                           PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})