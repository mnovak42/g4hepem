
#ifndef G4HepEmElectronInteractionBremSB_HH
#define G4HepEmElectronInteractionBremSB_HH

class  G4HepEmTLData;
struct G4HepEmData;
struct G4HepEmElectronData;


// Bremsstrahlung interaction based on the numerical Seltzer-Berger DCS for the 
// emitted photon energy. 
// Used between 100 eV - 1 GeV primary e-/e+ kinetic energies.
// NOTE: the core part i.e. sampling the emitted photon energy is different than 
//       that in the G4SeltzerBergerModel. I implemented here my rejection free,
//       memory effcicient (tables only per Z and not per mat-cuts) sampling. 
//       Rejection is used only to account dielectric supression and e+ correction. 
void PerformElectronBremSB(G4HepEmTLData* tlData, struct G4HepEmData* hepEmData, bool iselectron);

// Bremsstrahlung interaction based on the Bethe-Heitler DCS with several, but 
// most importantly, with LPM correction. 
// Used between 1 GeV - 100 TeV primary e-/e+ kinetic energies.
//void PerformElectronBremRB(G4HepEmTLData* tlData, struct G4HepEmData* hepEmData, bool iselectron);



// Target atom selector for the above bremsstrahlung intercations in case of 
// materials composed from multiple elements.
int SelectTargetAtomBrem(struct G4HepEmElectronData* elData, int imc, double ekin, double lekin, double urndn, bool isbremSB);

// Simple linear search (with step of 3!) used in the photon energy sampling part 
// of the SB (Seltzer-Berger) brem model.
int LinSearch(const double* vect, const int size, const double val);


#endif // G4HepEmElectronInteractionBremSB_HH