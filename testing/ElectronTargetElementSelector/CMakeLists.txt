add_executable(TestElemSelectorData
  TestElemSelectorData.cc
  src/Implementation.cc
  $<$<BOOL:${G4HepEm_CUDA_BUILD}>:src/ElemSelector.cu>)

target_include_directories(TestElemSelectorData PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(TestElemSelectorData
  PRIVATE
  g4HepEm TestUtils ${Geant4_LIBRARIES})

add_test(NAME TestElemSelectorData COMMAND TestElemSelectorData)