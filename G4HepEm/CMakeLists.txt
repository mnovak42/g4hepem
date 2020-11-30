

add_subdirectory (G4HepEmData)
add_subdirectory (G4HepEmInit)
add_subdirectory (G4HepEmRun)


include_directories (
 ${CMAKE_SOURCE_DIR}/G4HepEm/include
 ${G4HEPEMDATA_INC_DIR}
 ${G4HEPEMINIT_INC_DIR}
 ${G4HEPEMRUN_INC_DIR}
)

message (STATUS "??? ${G4HEPEMINIT_INC_DIR}")

file ( GLOB G4HEPEM_headers ${CMAKE_SOURCE_DIR}/G4HepEm/include/*.hh)
file ( GLOB G4HEPEM_sources ${CMAKE_SOURCE_DIR}/G4HepEm/src/*.cc)
if (BUILD_STATIC_LIBS)
  add_library (g4HepEm STATIC ${G4HEPEM_sources})
else ()
  add_library (g4HepEm SHARED ${G4HEPEM_sources})
endif ()

set_target_properties(g4HepEm PROPERTIES COMPILE_FLAGS ${CMAKE_CXX_FLAGS})
target_link_libraries(g4HepEm g4HepEmData g4HepEmInit g4HepEmRun) # ${Geant4_LIBRARIES})

## ----------------------------------------------------------------------------
## Install G4HepEm libraries and headers
install (FILES ${G4HEPEM_headers} DESTINATION includes)
install (TARGETS g4HepEm DESTINATION lib)

## ----------------------------------------------------------------------------
## Create and install G4HepEm CMake Config file
##
configure_file (${CMAKE_SOURCE_DIR}/cmake/G4HepEmConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/G4HepEmConfig.cmake @ONLY)

install (FILES
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/G4HepEmConfig.cmake
    DESTINATION lib/cmake/G4HepEm
)



###############################################################################
##  Set some global properties for retrieving proper configuration at testing 
###############################################################################

# local directory in the build for the header files
FILE ( GLOB G4HEPEMDATA_HEADERS "${G4HEPEMDATA_INC_DIR}/*")
FILE ( GLOB G4HEPEMINIT_HEADERS "${G4HEPEMINIT_INC_DIR}/*")
FILE ( GLOB G4HEPEMRUN_HEADERS "${G4HEPEMRUN_INC_DIR}/*")
FILE ( COPY ${G4HEPEM_headers} ${G4HEPEMDATA_HEADERS} ${G4HEPEMINIT_HEADERS} ${G4HEPEMRUN_HEADERS} DESTINATION ${CMAKE_BINARY_DIR}/includes)
set (G4HEPEM_TESTING_INCLUDE_DIR "${CMAKE_BINARY_DIR}/includes")

# set G4HepEm include directory property used (only) property for testing 
set_property ( GLOBAL PROPERTY G4HEPEM_TESTING_INCLUDE_DIR_PROP  "${G4HEPEM_TESTING_INCLUDE_DIR}" )
