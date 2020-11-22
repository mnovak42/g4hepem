#----------------------------------------------------------------------------
# Setup the project
cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project (TestElemSelectorData)

#-------------------------------------------------------------------------------
# Set the CXX flags that were used when building the library
#
set ( CMAKE_CXX_FLAGS "${G4HepEm_CMAKE_CXX_FLAGS}" )

file ( GLOB headers     ${CMAKE_SOURCE_DIR}/include/*.hh)
file ( GLOB cxx_sources ${CMAKE_SOURCE_DIR}/src/*.cc)
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
  add_executable ( TestElemSelectorData ${CMAKE_SOURCE_DIR}/TestElemSelectorData.cc ${cxx_sources} ${cu_sources})  
  target_compile_options (TestElemSelectorData PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                         -Xcompiler -fPIC -Xcudafe 
                         --diag_suppress=boolean_controlling_expr_is_constant
                         --disable-warnings
                         -lineinfo
                         >)
  set_property ( TARGET TestElemSelectorData PROPERTY CUDA_SEPARABLE_COMPILATION ON )
  target_link_libraries( TestElemSelectorData ${Geant4_LIBRARIES} ${G4HepEm_LIBRARIES} )  
else ( G4HepEm_CUDA_BUILD )
  add_executable ( TestElemSelectorData ${CMAKE_SOURCE_DIR}/TestElemSelectorData.cc ${cxx_sources})
  set_target_properties (TestElemSelectorData  PROPERTIES COMPILE_FLAGS ${G4HepEm_CMAKE_CXX_FLAGS})
  target_link_libraries (TestElemSelectorData  ${Geant4_LIBRARIES} ${G4HepEm_LIBRARIES})  
endif ( G4HepEm_CUDA_BUILD )


