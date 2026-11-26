add_executable(TestElemSelectorData
  TestElemSelectorData.cc
  src/Implementation.cc)
g4hepem_add_device_test_sources(TestElemSelectorData src/ElemSelector.cu)

target_include_directories(TestElemSelectorData PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(TestElemSelectorData
  PRIVATE
  g4HepEm TestUtils ${Geant4_LIBRARIES})

add_test(NAME TestElemSelectorData COMMAND TestElemSelectorData)