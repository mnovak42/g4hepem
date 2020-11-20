

set (G4HEPEMDATA_INC_DIR ${CMAKE_SOURCE_DIR}/G4HepEm/G4HepEmData/include PARENT_SCOPE)

include_directories (${CMAKE_SOURCE_DIR}/G4HepEm/G4HepEmData/include)
file ( GLOB G4HEPEMDATA_headers ${CMAKE_SOURCE_DIR}/G4HepEm/G4HepEmData/include/*.hh)
file ( GLOB G4HEPEMDATA_CXX_sources ${CMAKE_SOURCE_DIR}/G4HepEm/G4HepEmData/src/*.cc)

#
# CUDA

if (G4HepEm_CUDA_BUILD)
  file (GLOB G4HEPEMDATA_CUDA_sources ${CMAKE_SOURCE_DIR}/G4HepEm/G4HepEmData/src/*.cu)
  # add possible CUDA headers
  # list (APPEND G4HEPEMDATA_headers )
  #
  if (BUILD_STATIC_LIBS)
    cuda_add_library (g4HepEmData STATIC ${G4HEPEMDATA_CXX_sources} ${G4HEPEMDATA_CUDA_sources})
  else (BUILD_STATIC_LIBS)
    cuda_add_library (g4HepEmData SHARED ${G4HEPEMDATA_CXX_sources} ${G4HEPEMDATA_CUDA_sources})
  endif (BUILD_STATIC_LIBS)
  set_property(TARGET g4HepEmData PROPERTY CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(g4HepEmData ${CUDA_LIBRARIES})
else (G4HepEm_CUDA_BUILD)
  if (BUILD_STATIC_LIBS)
    add_library (g4HepEmData STATIC ${G4HEPEMDATA_CXX_sources})
  else (BUILD_STATIC_LIBS)
    add_library (g4HepEmData SHARED ${G4HEPEMDATA_CXX_sources})
  endif (BUILD_STATIC_LIBS)
  set_target_properties(g4HepEmData PROPERTIES COMPILE_FLAGS ${CMAKE_CXX_FLAGS})
endif (G4HepEm_CUDA_BUILD)


## ----------------------------------------------------------------------------
## Install G4HepEm libraries and headers
install (FILES ${G4HEPEMDATA_headers} DESTINATION includes)
install (TARGETS g4HepEmData DESTINATION lib)


