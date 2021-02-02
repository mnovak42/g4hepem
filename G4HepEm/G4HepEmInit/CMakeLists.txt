file(GLOB G4HEPEMInit_headers ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hh)
file(GLOB G4HEPEMInit_sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc)

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
