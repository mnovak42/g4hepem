## ----------------------------------------------------------------------------
## 1. TestUtils helper library
##
add_library(TestUtils STATIC
  TestUtils/G4HepEmDataComparison.hh
  TestUtils/G4SetUp.hh
  TestUtils/G4SetUp.cc
  TestUtils/Hist.hh)
target_compile_features(TestUtils PUBLIC cxx_std_${CMAKE_CXX_STANDARD})
target_include_directories(TestUtils PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(TestUtils PUBLIC ${Geant4_LIBRARIES})

## Device drivers of the data tests: CUDA sources, compiled as HIP in HIP builds
if(G4HepEm_HIP_BUILD)
  if(CMAKE_VERSION VERSION_LESS 3.21)
    message(FATAL_ERROR "The HIP device tests need CMake 3.21 or later (HIP language support)")
  endif()
  enable_language(HIP)
endif()
function(g4hepem_add_device_test_sources target)
  if(G4HepEm_CUDA_BUILD OR G4HepEm_HIP_BUILD)
    target_sources(${target} PRIVATE ${ARGN})
    if(G4HepEm_HIP_BUILD)
      set_source_files_properties(${ARGN} PROPERTIES LANGUAGE HIP)
    endif()
  endif()
endfunction()

## ----------------------------------------------------------------------------
## 2. Add the unit test applications
##
find_package(GTest QUIET)
if(GTest_FOUND)
  add_subdirectory(G4HepEmDataInterfaces)
else()
  message(STATUS "Disabling G4HepEmDataInterfaces: GTest not found")
endif()

add_subdirectory(ElectronEnergyLoss)
add_subdirectory(ElectronTargetElementSelector)
add_subdirectory(GammaTargetElementSelector)
add_subdirectory(ElectronXSections)
add_subdirectory(GammaXSections)
add_subdirectory(MaterialAndRelated)
add_subdirectory(DataImportExport)
add_subdirectory(DataInitialization)

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
