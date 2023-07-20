# - Build component libraries
add_subdirectory(G4HepEmData)
add_subdirectory(G4HepEmDataJsonIO)
add_subdirectory(G4HepEmRun)

if(G4HepEm_GEANT4_BUILD)
  add_subdirectory(G4HepEmInit)
  add_subdirectory(G4HepEm)
endif()

