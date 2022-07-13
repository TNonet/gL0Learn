cmake_minimum_required(VERSION 3.4...3.18)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

IF (NOT DEFINED GL0LEARN_VERSION_INFO)
    SET(GL0LEARN_VERSION_INFO "0.0.0")
ENDIF ()

SET(MODNAME "gl0learn")

PROJECT(
        ${MODNAME}
        VERSION ${GL0LEARN_VERSION_INFO}
        DESCRIPTION "Python bindings for gL0Learn CD and CDPSI optimization."
        LANGUAGES CXX
)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "build shared library" FORCE)

ADD_SUBDIRECTORY(external/pybind11)
ADD_SUBDIRECTORY(external/armadillo-code)
ADD_SUBDIRECTORY(external/carma)

# Add Armadillo and pybind11
if($ENV{CIBUILDWHEEL})
    set(ALLOW_OPENBLAS_MACOS ON CACHE BOOL "Allow detection of OpenBLAS on macOS" FORCE)
    target_compile_definitions(armadillo PRIVATE -DARMA_USE_LAPACK)
    target_compile_definitions(armadillo PRIVATE -DARMA_USE_BLAS)
endif()

INCLUDE_DIRECTORIES(gl0learn)
INCLUDE_DIRECTORIES(gl0learn/src/include)
INCLUDE_DIRECTORIES(gl0learn/src/src/include)

pybind11_add_module(${MODNAME} MODULE
        gl0learn/src/pygl0learn.cpp
        gl0learn/src/pyoracle.cpp
        gl0learn/src/pyfitmodel.cpp
        gl0learn/src/pyfit.cpp)



TARGET_LINK_LIBRARIES(${MODNAME} PRIVATE carma::carma)
TARGET_LINK_LIBRARIES(${MODNAME} PRIVATE armadillo)


# Add -fPIC for Armadillo (and OpenBLAS if compiled)
if(NOT MSVC)
    # clang on Windows does not support -fPIC
    if(NOT WIN32)
        target_compile_options(armadillo PRIVATE -fPIC)
    endif()
endif()

TARGET_INCLUDE_DIRECTORIES(${MODNAME}
        PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        )

TARGET_COMPILE_OPTIONS(${MODNAME}
        PUBLIC
        "$<$<CONFIG:RELEASE>:${PROJECT_RELEASE_FLAGS}>"
        )

TARGET_COMPILE_DEFINITIONS(${MODNAME}
        PUBLIC
        "$<$<CONFIG:RELEASE>:${PROJECT_RELEASE_DEFINITIONS}>"
        )


INSTALL(TARGETS ${MODNAME} DESTINATION .)
