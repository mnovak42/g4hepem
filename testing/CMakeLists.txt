## ----------------------------------------------------------------------------
## 1. TestUtils helper library
##
add_library(TestUtils STATIC
  TestUtils/G4SetUp.hh
  TestUtils/G4SetUp.cc
  TestUtils/Hist.hh)
target_compile_features(TestUtils PUBLIC cxx_std_${CMAKE_CXX_STANDARD})
target_include_directories(TestUtils PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(TestUtils PUBLIC ${Geant4_LIBRARIES})

## ----------------------------------------------------------------------------
## 2. Add the unit test applications
##
add_subdirectory(ElectronEnergyLoss)
add_subdirectory(ElectronTargetElementSelector)
add_subdirectory(ElectronXSections)
add_subdirectory(MaterialAndRelated)

## ----------------------------------------------------------------------------
## 3. Add the example applications as tests
##
set(G4HepEm_DIR "${PROJECT_BINARY_DIR}")
add_subdirectory(${PROJECT_SOURCE_DIR}/apps/examples/TestEm3 ${CMAKE_CURRENT_BINARY_DIR}/TestEm3)
add_test(NAME TestEm3 COMMAND TestEm3 -m "${PROJECT_SOURCE_DIR}/apps/examples/TestEm3/ATLASbar.mac")

