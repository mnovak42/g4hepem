# Testing Export and Import of G4HepEM data

This test constructs a mock Geant4 setup consisting of two materials (Pb and LAr)
and a nested geometry of three volumes. Two `G4Regions` are defined with different
production cuts. After setting up this system, the `G4HepEm` data tables are constructed
(Host only, no Device side operations are needed in this test) and written out to file
together with the geometry information in a separate GDML file.