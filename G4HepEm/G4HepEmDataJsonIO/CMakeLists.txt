set(G4HEPEMDATAJSONIO_headers
  include/G4HepEmDataJsonIO.hh
)
set(G4HEPEMDATAJSONIO_CXX_sources
  src/G4HepEmDataJsonIO.cc
)

g4hepem_add_library(g4HepEmDataJsonIO
  SOURCES ${G4HEPEMDATAJSONIO_CXX_sources}
  HEADERS ${G4HEPEMDATAJSONIO_headers}
  LINK g4HepEmData)
