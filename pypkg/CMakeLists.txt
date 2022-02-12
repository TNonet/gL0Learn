cmake_minimum_required(VERSION 3.4...3.18)
project(gl0learn)

add_subdirectory(external/pybind11)
add_subdirectory(external/carma)

pybind11_add_module(gl0learn SHARED
    gl0learn/pygl0learn.cpp
)

target_include_directories(gl0learn PRIVATE gl0learn/src/include)

TARGET_LINK_LIBRARIES(gl0learn PUBLIC carma)

TARGET_INCLUDE_DIRECTORIES(gl0learn
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

TARGET_COMPILE_OPTIONS(gl0learn
    PUBLIC
        "$<$<CONFIG:RELEASE>:${PROJECT_RELEASE_FLAGS}>"
)

TARGET_COMPILE_DEFINITIONS(gl0learn
    PUBLIC
        "$<$<CONFIG:RELEASE>:${PROJECT_RELEASE_DEFINITIONS}>"
)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(gl0learn
                           PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})