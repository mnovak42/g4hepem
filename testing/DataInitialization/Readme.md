# Testing initialization via G4HepEm and G4HepEmInit

G4HepEm's data structures (`G4HepEmParameters`, `G4HepEmData` and members) can be
initialized using the functions declared in `G4HepEmInit`. These are also called
by G4HepEm's `G4HepEmRunManager.hh` for use in CPU Geant4.

This test confirms that the data structures constructed by both methods are numerically
identical.
