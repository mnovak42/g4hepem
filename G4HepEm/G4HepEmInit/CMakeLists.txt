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

if(BUILD_STATIC_LIBS)
  add_library(g4HepEmInit STATIC ${G4HEPEMInit_sources})
else()
  add_library(g4HepEmInit SHARED ${G4HEPEMInit_sources})
endif()

add_library(${PROJECT_NAME}::g4HepEmInit ALIAS g4HepEmInit)

target_compile_features(g4HepEmInit PUBLIC cxx_std_${CMAKE_CXX_STANDARD})
target_include_directories(g4HepEmInit PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}>)
target_link_libraries(g4HepEmInit PUBLIC g4HepEmData ${Geant4_LIBRARIES})

## ----------------------------------------------------------------------------
## Install G4HepEm libraries and headers
install(FILES ${G4HEPEMInit_headers} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}")
install(TARGETS g4HepEmInit
  EXPORT ${PROJECT_NAME}Targets
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
