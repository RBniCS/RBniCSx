cmake_minimum_required(VERSION 3.18)

project(rbnicsx)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set default build type
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Find python
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Find nanobind
execute_process(
    COMMAND ${Python_EXECUTABLE} -m nanobind --cmake_dir
    OUTPUT_VARIABLE NANOBIND_CMAKE_DIR
    RESULT_VARIABLE NANOBIND_CMAKE_DIR_COMMAND_RESULT
    ERROR_VARIABLE NANOBIND_CMAKE_DIR_COMMAND_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT NANOBIND_CMAKE_DIR_COMMAND_RESULT)
    list(APPEND CMAKE_PREFIX_PATH "${NANOBIND_CMAKE_DIR}")
    find_package(nanobind CONFIG REQUIRED)
    message(STATUS "Found nanobind python wrappers at ${NANOBIND_CMAKE_DIR}")
else()
    message(FATAL_ERROR "nanobind could not be found.")
endif()

# Check for PETSc
find_package(PkgConfig REQUIRED)
set(ENV{PKG_CONFIG_PATH}
  "$ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/lib/pkgconfig:$ENV{PETSC_DIR}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}"
)
pkg_search_module(PETSC REQUIRED IMPORTED_TARGET PETSc>=3.15 petsc>=3.15)

# Check for petsc4py
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import petsc4py; print(petsc4py.get_include())"
    OUTPUT_VARIABLE PETSC4PY_INCLUDE_DIR
    RESULT_VARIABLE PETSC4PY_INCLUDE_COMMAND_RESULT
    ERROR_VARIABLE PETSC4PY_INCLUDE_COMMAND_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT PETSC4PY_INCLUDE_COMMAND_RESULT)
    message(STATUS "Found petsc4py include directory at ${PETSC4PY_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "petsc4py could not be found.")
endif()

# Compile rbnicsx C++ backend and nanobind wrappers
nanobind_add_module(
    rbnicsx_cpp
    MODULE
    rbnicsx/_backends/frobenius_inner_product.cpp
    rbnicsx/_backends/petsc_error.cpp
    rbnicsx/wrappers/_backends.cpp
    rbnicsx/wrappers/rbnicsx.cpp
)

# Add current source directory to include directories
target_include_directories(rbnicsx_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Add PETSc and petsc4py include directories
target_include_directories(rbnicsx_cpp PRIVATE ${PETSC_INCLUDE_DIRS})
target_include_directories(rbnicsx_cpp PRIVATE ${PETSC4PY_INCLUDE_DIR})

# Add PETSc libraries
target_link_libraries(rbnicsx_cpp PRIVATE PkgConfig::PETSC)

# Install the compiled library to the _cpp subdirectory
install(TARGETS rbnicsx_cpp LIBRARY DESTINATION rbnicsx/_cpp)
