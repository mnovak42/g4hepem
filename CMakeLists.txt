#----------------------------------------------------------------------------
# Setup the project; take policies from up to CMake 3.18 to avoid warnings
# concerning CMP0104.
cmake_minimum_required(VERSION 3.17...3.18 FATAL_ERROR)
project(G4HepEm VERSION 0.1.0)

#----------------------------------------------------------------------------
# Option for enabling early exit
option(G4HepEm_EARLY_TRACKING_EXIT "Enable user-defined early tracking exit" OFF)
if(G4HepEm_EARLY_TRACKING_EXIT)
  message(STATUS "User-defined early tracking exit is enabled")
endif()

# Local and Core Modules
include(GNUInstallDirs)
include(CheckLanguage)

#----------------------------------------------------------------------------
# Convenience function to add shared / static libraries as configured.
#
# The general idea is as follows:
#  - if configured, shared libraries are created from a target called ${_name}.
#  - if configured, static libraries are created from a target always called
#    ${_name}-static but the final archive name is lib${_name}.a.
#  - if only static libraries are configured, ALIAS targets ${_name} are added
#    and exported.
function(g4hepem_add_library _name)
  cmake_parse_arguments(_g4hepem "" "" "HEADERS;SOURCES;LINK" ${ARGN})

  # Build shared library, if enabled.
  if(BUILD_SHARED_LIBS)
    add_library(${_name} SHARED ${_g4hepem_SOURCES})
    target_compile_features(${_name} PUBLIC cxx_std_${CMAKE_CXX_STANDARD})
    add_library(${PROJECT_NAME}::${_name} ALIAS ${_name})

    target_include_directories(${_name} PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}>)
    target_link_libraries(${_name} PUBLIC ${_g4hepem_LINK})

    if(G4HepEm_EARLY_TRACKING_EXIT)
      target_compile_definitions(${_name} PUBLIC G4HepEm_EARLY_TRACKING_EXIT)
    endif()
  endif()

  # Build static library, if enabled.
  if(BUILD_STATIC_LIBS)
    add_library(${_name}-static STATIC ${_g4hepem_SOURCES})
    set_target_properties(${_name}-static PROPERTIES OUTPUT_NAME ${_name})
    target_compile_features(${_name}-static PUBLIC cxx_std_${CMAKE_CXX_STANDARD})
    add_library(${PROJECT_NAME}::${_name}-static ALIAS ${_name}-static)

    target_include_directories(${_name}-static PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}>)
    foreach(_lib ${_g4hepem_LINK})
      target_link_libraries(${_name}-static PUBLIC ${_lib}-static)
    endforeach()

    if(G4HepEm_EARLY_TRACKING_EXIT)
      target_compile_definitions(${_name}-static PUBLIC G4HepEm_EARLY_TRACKING_EXIT)
    endif()

    # If only the static library, add alias targets for convenience.
    if(NOT BUILD_SHARED_LIBS)
      add_library(${_name} ALIAS ${_name}-static)
      add_library(${PROJECT_NAME}::${_name} ALIAS ${_name}-static)

      # In that case, also change the name of the exported target.
      set_target_properties(${_name}-static PROPERTIES EXPORT_NAME ${_name})
    endif()
  endif()

  # Install headers.
  install(FILES ${_g4hepem_HEADERS} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}")

  # Install libraries, as built.
  if(BUILD_SHARED_LIBS)
    install(TARGETS ${_name}
      EXPORT ${PROJECT_NAME}Targets
      ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
      LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
      RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
  endif()

  if(BUILD_STATIC_LIBS)
    install(TARGETS ${_name}-static
      EXPORT ${PROJECT_NAME}Targets
      ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
      LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
      RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
  endif()
endfunction()

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

# - Prefer shared libs
set(BUILD_SHARED_LIBS ON)
set(BUILD_STATIC_LIBS OFF)

# - Single location for library build products
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

#----------------------------------------------------------------------------
# Find Geant4, changing library build type to default Geant4 variant and
# determining which CXX Standard and CLHEP target to use
option(G4HepEm_GEANT4_BUILD "Build with Geant4 runtime support" ON)
if(G4HepEm_GEANT4_BUILD)
  find_package(Geant4 10.6 REQUIRED)
  if(Geant4_static_FOUND)
    set(BUILD_STATIC_LIBS ON)
  endif()
  if(NOT Geant4_shared_FOUND)
    set(BUILD_SHARED_LIBS OFF)
  endif()

  set(CMAKE_CXX_STANDARD ${Geant4_CXX_STANDARD}) # use value from Geant4Config.cmake
else()
  # Need workaround for G4VERSION_NUM G4HepEmRun...
  # ... done in that component for now.
endif()

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 11)
  message(WARNING "Find of Geant4 did not set Geant4_CXX_STANDARD value expected from Geant4Config.cmake. Defaulting to C++11.")
endif()

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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
  if(G4HepEm_GEANT4_BUILD)
    message(STATUS "Building test programs is enabled!")
    add_subdirectory(testing)
  else()
    message(WARNING "BUILD_TESTING is ON, but no tests will be created as Geant4 runtime support is disabled")
  endif()
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
