add_executable(testBrem
  testBrem.cc
  src/Implementation.cc
  src/TestBremArgs.cc
  $<$<BOOL:${G4HepEm_CUDA_BUILD}>:src/SBTableData.cu>)

target_include_directories(testBrem PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(testBrem
  PRIVATE
  g4HepEm TestUtils ${Geant4_LIBRARIES})

add_test(NAME TestBrem-Interaction COMMAND testBrem)
