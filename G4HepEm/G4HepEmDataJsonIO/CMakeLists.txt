file(GLOB G4HEPEMDATAJSONIO_headers ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hh)
file(GLOB G4HEPEMDATAJSONIO_CXX_sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc)

if(BUILD_STATIC_LIBS)
  add_library(g4HepEmDataJsonIO STATIC ${G4HEPEMDATAJSONIO_CXX_sources})
else()
  add_library(g4HepEmDataJsonIO SHARED ${G4HEPEMDATAJSONIO_CXX_sources})
endif()

add_library(${PROJECT_NAME}::g4HepEmDataJsonIO ALIAS g4HepEmDataJsonIO)

target_compile_features(g4HepEmDataJsonIO PUBLIC cxx_std_${CMAKE_CXX_STANDARD})

target_include_directories(g4HepEmDataJsonIO PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}>
  PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)

target_link_libraries(g4HepEmDataJsonIO PUBLIC g4HepEmData)

## ----------------------------------------------------------------------------
## Install G4HepEm libraries and headers
install(FILES ${G4HEPEMDATAJSONIO_headers} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}")
install(TARGETS g4HepEmDataJsonIO
  EXPORT ${PROJECT_NAME}Targets
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
