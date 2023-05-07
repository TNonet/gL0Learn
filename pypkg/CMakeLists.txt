CMAKE_MINIMUM_REQUIRED(VERSION 3.16)

SET(USE_PYBIND11_VERSION "v2.6.2")
SET(USE_ARMA_VERSION 10.6.x)
SET(USE_CARMA_VERSION "v0.6.7")

SET(BUILD_SHARED_LIBS OFF CACHE BOOL "build shared library" FORCE)


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


pybind11_add_module(_hello MODULE
    src/hello/hello_py.cpp
)

TARGET_LINK_LIBRARIES(_hello PRIVATE carma::carma)
TARGET_LINK_LIBRARIES(_hello PRIVATE armadillo)


INSTALL(TARGETS _hello DESTINATION .)



