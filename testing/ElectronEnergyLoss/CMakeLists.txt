add_executable(TestEnergyLossData
  TestEnergyLossData.cc
  src/Implementation.cc)
g4hepem_add_device_test_sources(TestEnergyLossData src/ELossData.cu)

target_include_directories(TestEnergyLossData PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(TestEnergyLossData
  PRIVATE
  g4HepEm TestUtils ${Geant4_LIBRARIES})

add_test(NAME TestEnergyLossData COMMAND TestEnergyLossData)
