set(G4HEPEmRun_headers
  include/G4HepEmConstants.hh
  include/G4HepEmElectronInteractionBrem.hh
  include/G4HepEmElectronInteractionIoni.hh
  include/G4HepEmElectronInteractionMSC.hh
  include/G4HepEmElectronManager.hh
  include/G4HepEmElectronTrack.hh
  include/G4HepEmGammaInteractionCompton.hh
  include/G4HepEmGammaInteractionConversion.hh
  include/G4HepEmGammaInteractionPhotoelectric.hh
  include/G4HepEmGammaManager.hh
  include/G4HepEmGammaTrack.hh
  include/G4HepEmInteractionUtils.hh
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
  include/G4HepEmElectronInteractionBrem.icc
  include/G4HepEmElectronInteractionIoni.icc
  include/G4HepEmElectronInteractionMSC.icc
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
  src/G4HepEmTLData.cc
)

# See Issue #11 and discussion...
set_source_files_properties(${G4HEPEmRun_impl_headers} PROPERTIES LANGUAGE CXX)
if(BUILD_STATIC_LIBS)
  add_library(g4HepEmRun STATIC ${G4HEPEmRun_sources} ${G4HEPEmRun_impl_headers})
else()
  add_library(g4HepEmRun SHARED ${G4HEPEmRun_sources} ${G4HEPEmRun_impl_headers})
endif()

set_target_properties(g4HepEmRun PROPERTIES COMPILE_FLAGS "-x c++ ${CMAKE_CXX_FLAGS}")

add_library(${PROJECT_NAME}::g4HepEmRun ALIAS g4HepEmRun)

target_compile_features(g4HepEmRun PUBLIC cxx_std_${CMAKE_CXX_STANDARD})
target_include_directories(g4HepEmRun PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}>)
target_link_libraries(g4HepEmRun g4HepEmData ${G4HepEm_CLHEP_TARGET}) # only SystemOfUnits is used from CLHEP

## ----------------------------------------------------------------------------
## Install G4HepEm libraries and headers
install(FILES ${G4HEPEmRun_headers} ${G4HEPEmRun_impl_headers} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}")
install(TARGETS g4HepEmRun
  EXPORT ${PROJECT_NAME}Targets
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")

