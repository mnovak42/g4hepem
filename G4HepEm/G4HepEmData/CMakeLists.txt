file(GLOB G4HEPEMDATA_headers ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hh)
file(GLOB G4HEPEMDATA_CXX_sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc)

#
# CUDA
#
if (G4HepEm_CUDA_BUILD)
  file(GLOB G4HEPEMDATA_CUDA_sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu)
  # add possible CUDA headers
  # list (APPEND G4HEPEMDATA_headers )
  #
  if(BUILD_STATIC_LIBS)
    cuda_add_library(g4HepEmData STATIC ${G4HEPEMDATA_CXX_sources} ${G4HEPEMDATA_CUDA_sources})
  else()
    cuda_add_library(g4HepEmData SHARED ${G4HEPEMDATA_CXX_sources} ${G4HEPEMDATA_CUDA_sources})
  endif()
  set_property(TARGET g4HepEmData PROPERTY CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(g4HepEmData ${CUDA_LIBRARIES})
else()
  if(BUILD_STATIC_LIBS)
    add_library(g4HepEmData STATIC ${G4HEPEMDATA_CXX_sources})
  else()
    add_library(g4HepEmData SHARED ${G4HEPEMDATA_CXX_sources})
  endif()
endif()

add_library(${PROJECT_NAME}::g4HepEmData ALIAS g4HepEmData)

target_compile_features(g4HepEmData PUBLIC cxx_std_${CMAKE_CXX_STANDARD})
target_include_directories(g4HepEmData PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}>)

## ----------------------------------------------------------------------------
## Install G4HepEm libraries and headers
install(FILES ${G4HEPEMDATA_headers} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}")
install(TARGETS g4HepEmData
  EXPORT ${PROJECT_NAME}Targets
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
