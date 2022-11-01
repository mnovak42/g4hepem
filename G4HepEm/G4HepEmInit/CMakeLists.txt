set(G4HEPEMInit_headers
  include/G4HepEmElectronInit.hh
  include/G4HepEmElectronTableBuilder.hh
  include/G4HepEmGammaInit.hh
  include/G4HepEmGammaTableBuilder.hh
  include/G4HepEmInitUtils.hh
  include/G4HepEmMaterialInit.hh
  include/G4HepEmParametersInit.hh
  include/G4HepEmSBBremTableBuilder.hh
  include/G4HepEmStateInit.hh
)
set(G4HEPEMInit_sources
  src/G4HepEmElectronInit.cc
  src/G4HepEmElectronTableBuilder.cc
  src/G4HepEmGammaInit.cc
  src/G4HepEmGammaTableBuilder.cc
  src/G4HepEmInitUtils.cc
  src/G4HepEmMaterialInit.cc
  src/G4HepEmParametersInit.cc
  src/G4HepEmSBBremTableBuilder.cc
  src/G4HepEmStateInit.cc
)

set(G4HEPEMInit_Geant4_LIBRARIES
  Geant4::G4global
  Geant4::G4materials
  Geant4::G4particles
  Geant4::G4processes
)

g4hepem_add_library(g4HepEmInit
  SOURCES ${G4HEPEMInit_sources}
  HEADERS ${G4HEPEMInit_headers}
  LINK g4HepEmData ${G4HEPEMInit_Geant4_LIBRARIES})

if(BUILD_SHARED_LIBS)
  if(TARGET Geant4::G4zlib)
    target_link_libraries(g4HepEmInit PUBLIC Geant4::G4zlib)
  elseif(TARGET ZLIB::ZLIB)
    target_link_libraries(g4HepEmInit PUBLIC ZLIB::ZLIB)
  endif()
endif()
if(BUILD_STATIC_LIBS)
  if(TARGET Geant4::G4zlib-static)
    target_link_libraries(g4HepEmInit-static PUBLIC Geant4::G4zlib-static)
  elseif(TARGET ZLIB::ZLIB)
    # Not a typo, Geant4 also takes what it can get from zlib.
    target_link_libraries(g4HepEmInit-static PUBLIC ZLIB::ZLIB)
  endif()
endif()
