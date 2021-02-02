## ----------------------------------------------------------------------------
## Retrieve G4HepEm_INCLUDE_DIR (a collection of all headers set only for testing)
##
get_property(G4HepEm_INCLUDE_DIR GLOBAL PROPERTY G4HEPEM_TESTING_INCLUDE_DIR_PROP)

## ----------------------------------------------------------------------------
## List of unit tests: under the '/apps/tests/' directory
##
set(UNIT_TESTS_DIR_NAME
  "MaterialAndRelated"
  "ElectronEnergyLoss"
  "ElectronTargetElementSelector"
  "ElectronXSections")
set(UNIT_TESTS_NAME
  "TestMaterialAndRelated"
  "TestEnergyLossData"
  "TestElemSelectorData"
  "TestXSectionData")

## ----------------------------------------------------------------------------
## List of test applications: under the '/apps/examples' directory
##
set(TESTS
  "TestEm3")
set(TESTS_INPUT
  "-m ${PROJECT_SOURCE_DIR}/apps/examples/TestEm3/ATLASbar.mac")

## ----------------------------------------------------------------------------
## Inactivate GPU build for the testing (if it was active) since there is no DEVICE
##
if(NOT G4HepEm_CUDA_BUILD)
  remove_definitions(-DG4HepEm_CUDA_BUILD)
endif()


## ----------------------------------------------------------------------------
## First: the unit tests
##
list(LENGTH UNIT_TESTS_DIR_NAME num)
math(EXPR len "${num} - 1")
foreach(val RANGE ${len})
  list(GET UNIT_TESTS_DIR_NAME ${val}  t_DIR)
  list(GET UNIT_TESTS_NAME     ${val}  t_NAME)

  file(GLOB SOURCES ${PROJECT_SOURCE_DIR}/apps/tests/${t_DIR}/src/*.cc ${PROJECT_SOURCE_DIR}/apps/tests/TestUtils/src/*.cc)
  file(GLOB HEADERS ${PROJECT_SOURCE_DIR}/apps/tests/${t_DIR}/include/*.hh ${PROJECT_SOURCE_DIR}/apps/tests/TestUtils/include/*.hh)

  add_executable(${t_NAME} ${PROJECT_SOURCE_DIR}/apps/tests/${t_DIR}/${t_NAME}.cc ${SOURCES} ${HEADERS})
  target_include_directories(${t_NAME}
    PRIVATE
      "${PROJECT_SOURCE_DIR}/apps/tests/${t_DIR}/include"
      "${PROJECT_SOURCE_DIR}/apps/tests/TestUtils/include")
  target_link_libraries(${t_NAME} PRIVATE ${Geant4_LIBRARIES} g4HepEmData g4HepEmInit g4HepEmRun g4HepEm)

  add_test(${t_NAME} ${t_NAME})
endforeach ()


## ----------------------------------------------------------------------------
## Second: the test applications
##
list(LENGTH TESTS num)
math(EXPR len "${num} - 1")
foreach(val RANGE ${len})
  list(GET TESTS       ${val}  t_NAME)
  list(GET TESTS_INPUT ${val}  t_INPARS)

  file(GLOB SOURCES ${PROJECT_SOURCE_DIR}/apps/examples/${t_NAME}/src/*.cc)
  file(GLOB HEADERS ${PROJECT_SOURCE_DIR}/apps/examples/${t_NAME}/include/*.hh)

  add_executable(${t_NAME} ${PROJECT_SOURCE_DIR}/apps/examples/${t_NAME}/${t_NAME}.cc ${SOURCES} ${HEADERS})
  target_include_directories(${t_NAME} PRIVATE "${PROJECT_SOURCE_DIR}/apps/examples/${t_NAME}/include")
  target_link_libraries(${t_NAME} PRIVATE ${Geant4_LIBRARIES} g4HepEmData g4HepEmInit g4HepEmRun g4HepEm)

  add_test(${t_NAME} ${t_NAME} ${t_INPARS})
endforeach ()
