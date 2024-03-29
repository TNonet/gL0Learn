CMAKE_MINIMUM_REQUIRED(VERSION 3.16)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
SET(USE_PYBIND11_VERSION "v2.10.4")
SET(USE_ARMA_VERSION 10.6.x)
SET(USE_CARMA_VERSION "v0.6.7")

IF (NOT DEFINED GL0LEARN_VERSION_INFO)
    SET(GL0LEARN_VERSION_INFO "0.0.0")
ENDIF ()

PROJECT(
        gl0learn
        VERSION ${GL0LEARN_VERSION_INFO}
        DESCRIPTION "Python bindings for gL0Learn CD and CDPSI optimization."
        LANGUAGES CXX
)

SET(BUILD_SHARED_LIBS OFF CACHE BOOL "build shared library" FORCE)

SET(MODNAME _gl0learn)

# -- EXTERNAL
INCLUDE(GNUInstallDirs)
INCLUDE(FetchContent)

# Pybind11
FetchContent_Declare(
  Pybind11Repo
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        ${USE_PYBIND11_VERSION}
  SOURCE_DIR ${PROJECT_SOURCE_DIR}/extern/pybind11
)

FetchContent_GetProperties(Pybind11Repo)

STRING(TOLOWER "Pybind11Repo" lcName)
IF (NOT ${lcName}_POPULATED)
  MESSAGE(STATUS "gL0Learn: collecting Pybind11 ${USE_PYBIND11_VERSION}")
  # Fetch the content using previously declared details
  FetchContent_Populate(Pybind11Repo)
ENDIF ()

ADD_SUBDIRECTORY(extern/pybind11)

FetchContent_Declare(
  ArmadilloRepo
  GIT_REPOSITORY https://gitlab.com/conradsnicta/armadillo-code.git
  GIT_TAG        ${USE_ARMA_VERSION}
  SOURCE_DIR ${PROJECT_SOURCE_DIR}/extern/armadillo-code
)

FetchContent_GetProperties(ArmadilloRepo)

STRING(TOLOWER "ArmadilloRepo" lcName)
IF (NOT ${lcName}_POPULATED)
    MESSAGE(STATUS "gL0Learn: collecting Armadillo ${USE_ARMA_VERSION}")
    # Fetch the content using previously declared details
    FetchContent_Populate(ArmadilloRepo)
ENDIF ()

ADD_SUBDIRECTORY(extern/armadillo-code)

SET(ARMADILLO_INCLUDE_DIR extern/armadillo-code/include)

FetchContent_Declare(
  CarmaRepo
  GIT_REPOSITORY https://github.com/RUrlus/carma
  GIT_TAG        ${USE_CARMA_VERSION}
  SOURCE_DIR ${PROJECT_SOURCE_DIR}/extern/carma
)

FetchContent_GetProperties(CarmaRepo)

STRING(TOLOWER "CarmaRepo" lcName)
IF (NOT ${lcName}_POPULATED)
    MESSAGE(STATUS "gL0Learn: collecting carma ${USE_CARMA_VERSION}")
    # Fetch the content using previously declared details
    FetchContent_Populate(CarmaRepo)
ENDIF ()

ADD_SUBDIRECTORY(extern/carma)


INCLUDE_DIRECTORIES(src/gl0learn/src)
INCLUDE_DIRECTORIES(src/gl0learn/src/include)
INCLUDE_DIRECTORIES(src/gl0learn/src/src/include)

pybind11_add_module(${MODNAME} MODULE
    src/gl0learn/src/pygl0learn.cpp
    src/gl0learn/src/pyoracle.cpp
    src/gl0learn/src/pyfitmodel.cpp
    src/gl0learn/src/pyfit.cpp
)

# Add -fPIC for Armadillo (and OpenBLAS if compiled)
if(NOT MSVC)
    # clang on Windows does not support -fPIC
    if(NOT WIN32)
        target_compile_options(armadillo PRIVATE -fPIC)
    endif()
endif()

TARGET_LINK_LIBRARIES(${MODNAME} PRIVATE carma::carma)
TARGET_LINK_LIBRARIES(${MODNAME} PRIVATE armadillo)


INSTALL(TARGETS ${MODNAME} DESTINATION .)
