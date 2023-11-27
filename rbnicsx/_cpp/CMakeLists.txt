cmake_minimum_required(VERSION 3.18)

project(rbnicsx)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Report build type
message(STATUS "Build type is ${CMAKE_BUILD_TYPE}")

# Find python
find_package(
  Python
  COMPONENTS Interpreter Development
  REQUIRED
)

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

# Check for MPI
find_package(MPI 3 REQUIRED)

# Check for PETSc
find_package(PkgConfig REQUIRED)
set(ENV{PKG_CONFIG_PATH}
    "$ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/lib/pkgconfig:$ENV{PETSC_DIR}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}"
)
pkg_search_module(PETSC REQUIRED IMPORTED_TARGET PETSc>=3.15 petsc>=3.15)

# Check for petsc4py
execute_process(
  COMMAND ${Python_EXECUTABLE} -c
          "import petsc4py; print(petsc4py.get_include())"
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
  rbnicsx_cpp NOMINSIZE rbnicsx/_backends/frobenius_inner_product.cpp
  rbnicsx/_backends/petsc_error.cpp rbnicsx/wrappers/_backends.cpp
  rbnicsx/wrappers/rbnicsx.cpp
)

# Add MPI and PETSc libraries
target_link_libraries(rbnicsx_cpp PRIVATE MPI::MPI_CXX)
target_link_libraries(rbnicsx_cpp PRIVATE PkgConfig::PETSC)

# Add petsc4py include directories (with MPI and PETSc ones already being added
# by target_link_libraries)
target_include_directories(rbnicsx_cpp PRIVATE ${PETSC4PY_INCLUDE_DIR})

# Add current source directory to include directories
target_include_directories(rbnicsx_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Install the compiled library to the _cpp subdirectory
set_target_properties(rbnicsx_cpp PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
install(TARGETS rbnicsx_cpp LIBRARY DESTINATION rbnicsx/_cpp)
