add_executable(TestDataImportExport
  TestDataImportExport.cc
  include/MockG4.h
  src/MockG4.cc)

target_include_directories(TestDataImportExport PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(TestDataImportExport
  PRIVATE
  g4HepEm TestUtils ${Geant4_LIBRARIES})

add_test(NAME TestDataImportExport COMMAND TestDataImportExport)