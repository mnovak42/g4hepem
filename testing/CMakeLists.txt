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
add_subdirectory(GammaTargetElementSelector)
add_subdirectory(ElectronXSections)
add_subdirectory(GammaXSections)
add_subdirectory(MaterialAndRelated)
# Data test requires Geant4 GDML, so only add if supported
if(Geant4_gdml_FOUND)
  add_subdirectory(DataImportExport)
else()
  message(STATUS "Disabling TestDataImportExport: Geant4 GDML not found")
endif()

## ----------------------------------------------------------------------------
## 3. Add the developer-only test applications
##
add_subdirectory(testElectronInteractionBrem)

## ----------------------------------------------------------------------------
## 4. Add the example applications as tests
##
set(G4HepEm_DIR "${CMAKE_CURRENT_LIST_DIR}/shims")
add_subdirectory(${PROJECT_SOURCE_DIR}/apps/examples/TestEm3 ${CMAKE_CURRENT_BINARY_DIR}/TestEm3)
add_test(NAME TestEm3 COMMAND TestEm3 -m "${PROJECT_SOURCE_DIR}/apps/examples/TestEm3/ATLASbar.mac")
