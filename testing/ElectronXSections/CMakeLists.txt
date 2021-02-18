add_executable(TestXSectionData
  TestXSectionData.cc
  src/Implementation.cc
  $<$<BOOL:${G4HepEm_CUDA_BUILD}>:src/ResMacXSecs.cu>)

target_include_directories(TestXSectionData PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(TestXSectionData
  PRIVATE
  g4HepEm TestUtils ${Geant4_LIBRARIES})

add_test(NAME TestXSectionData COMMAND TestXSectionData)
