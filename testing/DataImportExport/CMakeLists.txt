add_executable(TestDataImportExport
  TestDataImportExport.cc
  include/SimpleFakeG4Setup.h
  src/SimpleFakeG4Setup.cc)

target_include_directories(TestDataImportExport PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(TestDataImportExport
  PRIVATE
  g4HepEm g4HepEmDataJsonIO TestUtils ${Geant4_LIBRARIES})

add_test(NAME TestDataImportExportFull COMMAND TestDataImportExport full)
add_test(NAME TestDataImportExportSimple COMMAND TestDataImportExport simple)
