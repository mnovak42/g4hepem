file(GLOB G4HEPEmRun_headers ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hh
                             ${CMAKE_CURRENT_SOURCE_DIR}/include/*.icc)
file(GLOB G4HEPEmRun_sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc
                             ${CMAKE_CURRENT_SOURCE_DIR}/include/*.icc)
# See Issue #11 and discussion...
set_source_files_properties(${G4HEPEmRun_sources} PROPERTIES LANGUAGE CXX)
if(BUILD_STATIC_LIBS)
  add_library(g4HepEmRun STATIC ${G4HEPEmRun_sources})
else()
  add_library(g4HepEmRun SHARED ${G4HEPEmRun_sources})
endif()

set_target_properties(g4HepEmRun PROPERTIES COMPILE_FLAGS "-x c++ ${CMAKE_CXX_FLAGS}")

add_library(${PROJECT_NAME}::g4HepEmRun ALIAS g4HepEmRun)

target_compile_features(g4HepEmRun PUBLIC cxx_std_${CMAKE_CXX_STANDARD})
target_include_directories(g4HepEmRun PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}>)
target_link_libraries(g4HepEmRun g4HepEmData ${G4HepEm_CLHEP_TARGET}) # only rng is used from G4 at run

## ----------------------------------------------------------------------------
## Install G4HepEm libraries and headers
install(FILES ${G4HEPEmRun_headers} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}")
install(TARGETS g4HepEmRun
  EXPORT ${PROJECT_NAME}Targets
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")

