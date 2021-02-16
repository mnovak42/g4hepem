# Shimmed config file for G4HepEm to allow examples to be used as tests
# via a simple add_subdirectory
# It works round the chicken and egg problem of examples using
# find_package(G4HepEm) before G4HepEmConfig.cmake has been
# generated for the build tree. It simply provides the same interface
# as the full file.
set(G4HepEm_LIBRARIES g4HepEm g4HepEmRun g4HepEmInit g4HepEmData)
