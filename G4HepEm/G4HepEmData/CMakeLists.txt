file(GLOB G4HEPEMDATA_headers ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hh)
file(GLOB G4HEPEMDATA_CXX_sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc)

if(BUILD_STATIC_LIBS)
  add_library(g4HepEmData STATIC ${G4HEPEMDATA_CXX_sources})
else()
  add_library(g4HepEmData SHARED ${G4HEPEMDATA_CXX_sources})
endif()

add_library(${PROJECT_NAME}::g4HepEmData ALIAS g4HepEmData)

target_compile_definitions(g4HepEmData PUBLIC $<$<BOOL:${G4HepEm_CUDA_BUILD}>:G4HepEm_CUDA_BUILD>)

target_compile_features(g4HepEmData PUBLIC cxx_std_${CMAKE_CXX_STANDARD})

target_include_directories(g4HepEmData PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}>)

# TODO: Determine when/if to use CUDA::cudart_static
target_link_libraries(g4HepEmData PRIVATE $<$<BOOL:${G4HepEm_CUDA_BUILD}>:CUDA::cudart>)

## ----------------------------------------------------------------------------
## Install G4HepEm libraries and headers
install(FILES ${G4HEPEMDATA_headers} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}")
install(TARGETS g4HepEmData
  EXPORT ${PROJECT_NAME}Targets
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
