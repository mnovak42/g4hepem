
set (G4HEPEMRUN_INC_DIR ${CMAKE_SOURCE_DIR}/G4HepEm/G4HepEmRun/include PARENT_SCOPE)

include_directories (
  ${CMAKE_SOURCE_DIR}/G4HepEm/G4HepEmRun/include
  ${CMAKE_SOURCE_DIR}/G4HepEm/G4HepEmData/include
  ${Geant4_INCLUDE_DIRS}  # only for CLHEP:: Random and units/constants
  )
file ( GLOB G4HEPEmRun_headers ${CMAKE_SOURCE_DIR}/G4HepEm/G4HepEmRun/include/*.hh
                               ${CMAKE_SOURCE_DIR}/G4HepEm/G4HepEmRun/include/*.icc)
file ( GLOB G4HEPEmRun_sources ${CMAKE_SOURCE_DIR}/G4HepEm/G4HepEmRun/src/*.cc
                               ${CMAKE_SOURCE_DIR}/G4HepEm/G4HepEmRun/include/*.icc)
set_source_files_properties(${G4HEPEmRun_sources} PROPERTIES LANGUAGE CXX)
if (BUILD_STATIC_LIBS)
  add_library (g4HepEmRun STATIC ${G4HEPEmRun_sources})
else ()
  add_library (g4HepEmRun SHARED ${G4HEPEmRun_sources})
endif ()

set_target_properties(g4HepEmRun PROPERTIES COMPILE_FLAGS "-x c++ ${CMAKE_CXX_FLAGS}")
target_link_libraries(g4HepEmRun g4HepEmData Geant4::G4clhep) # only rng is used from G4 at run

## ----------------------------------------------------------------------------
## Install G4HepEm libraries and headers
install (FILES ${G4HEPEmRun_headers} DESTINATION includes)
install (TARGETS g4HepEmRun DESTINATION lib)
