set(G4HEPEmRun_headers
  include/G4HepEmConstants.hh
  include/G4HepEmElectronEnergyLossFluctuation.hh
  include/G4HepEmElectronInteractionBrem.hh
  include/G4HepEmElectronInteractionIoni.hh
  include/G4HepEmElectronInteractionUMSC.hh
  include/G4HepEmElectronManager.hh
  include/G4HepEmElectronTrack.hh
  include/G4HepEmExp.hh
  include/G4HepEmGammaInteractionCompton.hh
  include/G4HepEmGammaInteractionConversion.hh
  include/G4HepEmGammaInteractionPhotoelectric.hh
  include/G4HepEmGammaManager.hh
  include/G4HepEmGammaTrack.hh
  include/G4HepEmInteractionUtils.hh
  include/G4HepEmLog.hh
  include/G4HepEmMacros.hh
  include/G4HepEmMath.hh
  include/G4HepEmMSCTrackData.hh
  include/G4HepEmPositronInteractionAnnihilation.hh
  include/G4HepEmRandomEngine.hh
  include/G4HepEmRunUtils.hh
  include/G4HepEmTLData.hh
  include/G4HepEmTrack.hh
)
set(G4HEPEmRun_impl_headers
  include/G4HepEmElectronEnergyLossFluctuation.icc
  include/G4HepEmElectronInteractionBrem.icc
  include/G4HepEmElectronInteractionIoni.icc
  include/G4HepEmElectronInteractionUMSC.icc
  include/G4HepEmElectronManager.icc
  include/G4HepEmGammaInteractionCompton.icc
  include/G4HepEmGammaInteractionConversion.icc
  include/G4HepEmGammaInteractionPhotoelectric.icc
  include/G4HepEmGammaManager.icc
  include/G4HepEmInteractionUtils.icc
  include/G4HepEmPositronInteractionAnnihilation.icc
  include/G4HepEmRunUtils.icc
)
set(G4HEPEmRun_sources
  src/G4HepEmRandomEngine.cc
)

# See Issue #11 and discussion...
set_source_files_properties(${G4HEPEmRun_impl_headers} PROPERTIES LANGUAGE CXX)

g4hepem_add_library(g4HepEmRun
  SOURCES ${G4HEPEmRun_sources} ${G4HEPEmRun_impl_headers}
  HEADERS ${G4HEPEmRun_headers} ${G4HEPEmRun_impl_headers}
  LINK g4HepEmData)

# Make the Geant4 version number available even where we don't depend on Geant4
if(G4HepEm_GEANT4_BUILD)
  set(_g4version_num ${Geant4_VERSION_MAJOR}${Geant4_VERSION_MINOR}${Geant4_VERSION_PATCH})
else()
  set(_g4version_num 1100)
endif()

if(BUILD_SHARED_LIBS)
  set_target_properties(g4HepEmRun PROPERTIES COMPILE_FLAGS "-x c++ ${CMAKE_CXX_FLAGS}")

  target_compile_definitions(g4HepEmRun PUBLIC G4VERSION_NUM=${_g4version_num})

  if(TARGET Geant4::G4clhep)
    target_link_libraries(g4HepEmRun PUBLIC Geant4::G4clhep)
  elseif(TARGET CLHEP::CLHEP)
    target_link_libraries(g4HepEmRun PUBLIC CLHEP::CLHEP)
  endif()
endif()
if(BUILD_STATIC_LIBS)
  set_target_properties(g4HepEmRun-static PROPERTIES COMPILE_FLAGS "-x c++ ${CMAKE_CXX_FLAGS}")

  target_compile_definitions(g4HepEmRun-static PUBLIC G4VERSION_NUM=${_g4version_num})

  if(TARGET Geant4::G4clhep-static)
    target_link_libraries(g4HepEmRun-static PUBLIC Geant4::G4clhep-static)
  elseif(TARGET CLHEP::CLHEP)
    # Not a typo, Geant4 also takes what it can get from CLHEP.
    target_link_libraries(g4HepEmRun-static PUBLIC CLHEP::CLHEP)
  endif()
endif()
