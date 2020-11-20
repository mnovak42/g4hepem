
set (G4HEPEMINIT_INC_DIR ${CMAKE_SOURCE_DIR}/G4HepEm/G4HepEmInit/include PARENT_SCOPE)

include_directories (
   ${CMAKE_SOURCE_DIR}/G4HepEm/G4HepEmInit/include
   ${G4HEPEMDATA_INC_DIR}
)

file ( GLOB G4HEPEMInit_headers ${CMAKE_SOURCE_DIR}/G4HepEm/G4HepEmInit/include/*.hh)
file ( GLOB G4HEPEMInit_sources ${CMAKE_SOURCE_DIR}/G4HepEm/G4HepEmInit/src/*.cc)
if (BUILD_STATIC_LIBS)
  add_library (g4HepEmInit STATIC ${G4HEPEMInit_sources})
else ()
  add_library (g4HepEmInit SHARED ${G4HEPEMInit_sources})
endif ()

set_target_properties(g4HepEmInit PROPERTIES COMPILE_FLAGS ${CMAKE_CXX_FLAGS})
target_link_libraries(g4HepEmInit g4HepEmData ${Geant4_LIBRARIES})

## ----------------------------------------------------------------------------
## Install G4HepEm libraries and headers
install (FILES ${G4HEPEMInit_headers} DESTINATION includes)
install (TARGETS g4HepEmInit DESTINATION lib)


