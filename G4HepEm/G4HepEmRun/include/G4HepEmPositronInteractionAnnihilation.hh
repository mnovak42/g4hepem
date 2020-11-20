
#ifndef G4HepEmPositronInteractionAnnihilation_HH
#define G4HepEmPositronInteractionAnnihilation_HH

class  G4HepEmTLData;

// e+ annihilation to two gamma interaction described by the Heitler model.
// Used between 0 eV - 100 TeV primary e+ kinetic energies i.e. 
// covers both in-flight and at-rest annihilation.
void PerformPositronAnnihilation(G4HepEmTLData* tlData, bool isatrest);

// e+ is already at rest case
void AnnihilateAtRest(G4HepEmTLData* tlData);
// e+ is in-flight case
void AnnihilateInFlight(G4HepEmTLData* tlData);

#endif // G4HepEmPositronInteractionAnnihilation_HH