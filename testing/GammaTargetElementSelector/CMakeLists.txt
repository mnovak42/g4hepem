add_executable(TestGammaElemSelectorData
  TestGammaElemSelectorData.cc
  src/Implementation.cc
  $<$<BOOL:${G4HepEm_CUDA_BUILD}>:src/GammaElemSelector.cu>)

target_include_directories(TestGammaElemSelectorData PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(TestGammaElemSelectorData
  PRIVATE
  g4HepEm TestUtils ${Geant4_LIBRARIES})

add_test(NAME TestGammaElemSelectorData COMMAND TestGammaElemSelectorData)
