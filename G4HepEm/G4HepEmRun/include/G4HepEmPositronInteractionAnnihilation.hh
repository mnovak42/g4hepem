
#ifndef G4HepEmPositronInteractionAnnihilation_HH
#define G4HepEmPositronInteractionAnnihilation_HH

#include "G4HepEmMacros.hh"

class  G4HepEmTLData;
class  G4HepEmRandomEngine;

// e+ annihilation to two gamma interaction described by the Heitler model.
// Used between 0 eV - 100 TeV primary e+ kinetic energies i.e. 
// covers both in-flight and at-rest annihilation.
void PerformPositronAnnihilation(G4HepEmTLData* tlData, bool isatrest);

// e+ is already at rest case
void AnnihilateAtRest(G4HepEmTLData* tlData);
// e+ is in-flight case
void AnnihilateInFlight(G4HepEmTLData* tlData);

G4HepEmHostDevice
void SampleEnergyAndDirectionsForAnnihilationInFlight(const double thePrimEkin, const double *thePrimDir,
                                                      double *theGamma1Ekin, double *theGamma1Dir,
                                                      double *theGamma2Ekin, double *theGamma2Dir,
                                                      G4HepEmRandomEngine* rnge);

#endif // G4HepEmPositronInteractionAnnihilation_HH
