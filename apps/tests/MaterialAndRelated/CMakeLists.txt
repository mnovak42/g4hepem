#----------------------------------------------------------------------------
# Setup the project
cmake_minimum_required(VERSION 2.6...3.18 FATAL_ERROR)
project (TestMaterialAndRelated)

#-------------------------------------------------------------------------------
# Set the CXX flags that were used when building the library
#
set ( CMAKE_CXX_FLAGS "${G4HepEm_CMAKE_CXX_FLAGS}" )

file ( GLOB headers     ${CMAKE_SOURCE_DIR}/include/*.hh ${CMAKE_SOURCE_DIR}/../TestUtils/include/*.hh)
file ( GLOB cxx_sources ${CMAKE_SOURCE_DIR}/src/*.cc ${CMAKE_SOURCE_DIR}/../TestUtils/src/*.cc)
file ( GLOB cu_sources  ${CMAKE_SOURCE_DIR}/src/*.cu)

#----------------------------------------------------------------------------
# Find Geant4 package
#
find_package ( Geant4 REQUIRED )


#----------------------------------------------------------------------------
# Setup Geant4 include directories and compile definitions
#
include ( ${Geant4_USE_FILE} )


#----------------------------------------------------------------------------
# Find G4HepEm
#
find_package ( G4HepEm REQUIRED)


#----------------------------------------------------------------------------
# Locate sources and headers for this project
#
include_directories ( ${Geant4_INCLUDE_DIR}
                      ${G4HepEm_INCLUDE_DIR}
                      ${CMAKE_SOURCE_DIR}/include
                      ${CMAKE_SOURCE_DIR}/../TestUtils/include
                    )

#----------------------------------------------------------------------------
# Check if G4HepEm was built with CUDA support and enable testing if any
#
set ( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR} )

if ( G4HepEm_CUDA_BUILD )
message(STATUS " BUILDING WITH CUDA")
  enable_language ( CUDA )
  add_definitions ( -DG4HepEm_CUDA_BUILD )
  include_directories ( ${G4HepEm_CUDA_INCLUDE_DIR}
                        ${CMAKE_SOURCE_DIR}/include
                       )
  add_executable ( TestMaterialAndRelated ${CMAKE_SOURCE_DIR}/TestMaterialAndRelated.cc ${cxx_sources} ${cu_sources})
  target_link_libraries( TestMaterialAndRelated ${Geant4_LIBRARIES} ${G4HepEm_LIBRARIES} )
else ( G4HepEm_CUDA_BUILD )
  add_executable ( TestMaterialAndRelated ${CMAKE_SOURCE_DIR}/TestMaterialAndRelated.cc ${cxx_sources})
  set_target_properties (TestMaterialAndRelated  PROPERTIES COMPILE_FLAGS ${G4HepEm_CMAKE_CXX_FLAGS})
  target_link_libraries (TestMaterialAndRelated  ${Geant4_LIBRARIES} ${G4HepEm_LIBRARIES})
endif ( G4HepEm_CUDA_BUILD )
