#----------------------------------------------------------------------------
# Setup the project; take policies from up to CMake 3.18 to avoid warnings
# concerning CMP0104.
cmake_minimum_required(VERSION 3.17...3.18 FATAL_ERROR)
project(G4HepEm VERSION 0.1.0)

# Local and Core Modules
include(GNUInstallDirs)
include(CheckLanguage)

# CMake and Build Settings
# - Don't allow absolute paths other than CMAKE_INSTALL_PREFIX
set(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION ON)

# - Never export to or search in user/system package registry
set(CMAKE_EXPORT_NO_PACKAGE_REGISTRY ON)
set(CMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY ON)
set(CMAKE_FIND_PACKAGE_NO_SYSTEM_PACKAGE_REGISTRY ON)

# - Force project directories to appear first in any list of includes
set(CMAKE_INCLUDE_DIRECTORIES_PROJECT_BEFORE ON)

# - Only relink shared libs when interface changes
set(CMAKE_LINK_DEPENDS_NO_SHARED ON)

# - Only report newly installed files
set(CMAKE_INSTALL_MESSAGE LAZY)

# - Default to Release mode build in single mode
if(NOT CMAKE_CONFIGURATION_TYPES)
  if(NOT CMAKE_BUILD_TYPE)
    # Default to a Release build if nothing else...
    set(CMAKE_BUILD_TYPE Release)
  endif()
  # Force to the cache, but use existing value.
  set(CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}"
    CACHE STRING "Choose the type of build, options are: None Release Debug RelWithDebInfo MinSizeRel"
    FORCE)
endif()


# - Minimums required by G4HepEm
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# - Prefer shared libs
set(BUILD_SHARED_LIBS ON)
set(BUILD_STATIC_LIBS OFF)

# - Single location for library build products
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

#----------------------------------------------------------------------------
# Find Geant4, changing library build type to default Geant4 variant and
# determining which CLHEP target to use
find_package(Geant4 REQUIRED)
if(Geant4_static_FOUND AND NOT Geant4_shared_FOUND)
  set(BUILD_SHARED_LIBS OFF)
  set(BUILD_STATIC_LIBS ON)
endif()

set(G4HepEm_CLHEP_TARGET CLHEP::CLHEP)
if(TARGET Geant4::G4clhep)
  set(G4HepEm_CLHEP_TARGET Geant4::G4clhep)
endif()

#----------------------------------------------------------------------------
# CUDA
option(G4HepEm_CUDA_BUILD "BUILD with CUDA support" OFF)
if(G4HepEm_CUDA_BUILD)
  # Use host compiler by default to ensure ABI consistency
  set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}" CACHE STRING
    "Set to CMAKE_CXX_COMPILER by G4HepEM CMakeLists")

  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
  else()
    message(FATAL_ERROR "No CUDA compiler/support available")
  endif()

  # Find needed toolkit libraries
  find_package(CUDAToolkit REQUIRED)

  # On CMake < 3.18, need FindCUDA for Arch flags, and it's messy...
  if(CMAKE_VERSION VERSION_LESS 3.18)
    find_package(CUDA 10 REQUIRED QUIET)
    cuda_select_nvcc_arch_flags(G4HEPEM_CUDA_ARCH_FLAGS)
    string(REPLACE ";" " " G4HEPEM_CUDA_ARCH_FLAGS "${G4HEPEM_CUDA_ARCH_FLAGS}")
    set(CMAKE_CUDA_FLAGS "${G4HEPEM_CUDA_ARCH_FLAGS}")
  endif()

  set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)
endif()

#----------------------------------------------------------------------------
# Build G4HepEm libraries
add_subdirectory(G4HepEm)

## ----------------------------------------------------------------------------
## Add testing option, changing default from CTest to OFF
##
option(BUILD_TESTING "Build the testing tree." OFF)
include(CTest)
if(BUILD_TESTING)
  message(STATUS "Building test programs is enabled!")
  add_subdirectory(testing)
endif()

#-----------------------------------------------------------------------------
# Create/install support files
include(CMakePackageConfigHelpers)

# - Common
write_basic_package_version_file(
  ${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion)

# - Build Tree
set(G4HEPEM_EXPORTED_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/G4HepEm")

configure_package_config_file(cmake/G4HepEmConfig.cmake.in
  ${PROJECT_BINARY_DIR}/G4HepEmConfig.cmake
  INSTALL_PREFIX "${PROJECT_BINARY_DIR}"
  INSTALL_DESTINATION "${PROJECT_BINARY_DIR}"
  PATH_VARS G4HEPEM_EXPORTED_INCLUDE_DIR)

export(EXPORT ${PROJECT_NAME}Targets
  NAMESPACE ${PROJECT_NAME}::
  FILE ${PROJECT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake)

# - Install Tree
set(G4HEPEM_EXPORTED_INCLUDE_DIR "${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}")

configure_package_config_file(cmake/G4HepEmConfig.cmake.in
  ${PROJECT_BINARY_DIR}/InstallTree/G4HepEmConfig.cmake
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}"
  PATH_VARS G4HEPEM_EXPORTED_INCLUDE_DIR)

install(FILES ${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
              ${PROJECT_BINARY_DIR}/InstallTree/${PROJECT_NAME}Config.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})

install(EXPORT ${PROJECT_NAME}Targets
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})
