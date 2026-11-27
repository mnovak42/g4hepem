add_executable(TestMaterialAndRelated
  TestMaterialAndRelated.cc
  src/Implementation.cc)
g4hepem_add_device_test_sources(TestMaterialAndRelated src/ElementData.cu src/MatCutData.cu src/MaterialData.cu)

target_include_directories(TestMaterialAndRelated PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(TestMaterialAndRelated
  PRIVATE
  g4HepEm TestUtils ${Geant4_LIBRARIES})

add_test(NAME TestMaterialAndRelated COMMAND TestMaterialAndRelated)