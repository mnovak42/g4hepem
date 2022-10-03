# - Build component libraries
add_subdirectory(G4HepEmData)
add_subdirectory(G4HepEmDataJsonIO)
add_subdirectory(G4HepEmInit)
add_subdirectory(G4HepEmRun)

# - Now top level G4HepEm library
set(G4HEPEM_headers
  include/G4HepEmNoProcess.hh
  include/G4HepEmProcess.hh
  include/G4HepEmRunManager.hh
)
set(G4HEPEM_sources
  src/G4HepEmProcess.cc
  src/G4HepEmRunManager.cc
)

if(Geant4_VERSION VERSION_GREATER_EQUAL 11.0)
  set(G4HEPEM_headers ${G4HEPEM_headers}
    include/G4EmTrackingManager.hh
    include/HepEmTrackingManager.hh
  )
  set(G4HEPEM_sources ${G4HEPEM_sources}
    src/G4EmTrackingManager.cc
    src/HepEmTrackingManager.cc
  )
endif()

g4hepem_add_library(g4HepEm
  SOURCES ${G4HEPEM_sources}
  HEADERS ${G4HEPEM_headers}
  LINK g4HepEmData g4HepEmInit g4HepEmRun)
