add_executable(TestGammaXSectionData
  TestGammaXSectionData.cc
  src/Implementation.cc
  $<$<BOOL:${G4HepEm_CUDA_BUILD}>:src/GammaMacXSecs.cu>)

target_include_directories(TestGammaXSectionData PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(TestGammaXSectionData
  PRIVATE
  g4HepEm TestUtils ${Geant4_LIBRARIES})

add_test(NAME TestGammaXSectionData COMMAND TestGammaXSectionData)
