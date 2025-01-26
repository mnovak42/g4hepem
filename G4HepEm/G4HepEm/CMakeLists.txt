set(G4HEPEM_headers
  include/G4HepEmNoProcess.hh
  include/G4HepEmProcess.hh
  include/G4HepEmRunManager.hh
  include/G4HepEmWoodcockHelper.hh
)
set(G4HEPEM_sources
  src/G4HepEmProcess.cc
  src/G4HepEmRunManager.cc
  src/G4HepEmWoodcockHelper.cc
)

set(G4HEPEM_Geant4_LIBRARIES
  Geant4::G4global
  Geant4::G4processes
  Geant4::G4track
)

if(Geant4_VERSION VERSION_GREATER_EQUAL 11.0)
  set(G4HEPEM_headers ${G4HEPEM_headers}
    include/G4EmTrackingManager.hh
    include/G4HepEmConfig.hh
    include/G4HepEmTrackingManager.hh
  )
  set(G4HEPEM_sources ${G4HEPEM_sources}
    src/G4EmTrackingManager.cc
    src/G4HepEmConfig.cc
    src/G4HepEmTrackingManager.cc
  )

  set(G4HEPEM_Geant4_LIBRARIES ${G4HEPEM_Geant4_LIBRARIES}
    Geant4::G4event
    Geant4::G4tracking
  )
endif()

g4hepem_add_library(g4HepEm
  SOURCES ${G4HEPEM_sources}
  HEADERS ${G4HEPEM_headers}
  LINK g4HepEmData g4HepEmInit g4HepEmRun ${G4HEPEM_Geant4_LIBRARIES})
