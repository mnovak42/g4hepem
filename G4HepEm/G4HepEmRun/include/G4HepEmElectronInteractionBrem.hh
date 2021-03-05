
#ifndef G4HepEmElectronInteractionBrem_HH
#define G4HepEmElectronInteractionBrem_HH

#include "G4HepEmMacros.hh"

class  G4HepEmTLData;
class  G4HepEmRandomEngine;
struct G4HepEmData;
struct G4HepEmElectronData;


// Bremsstrahlung interaction based on:
// 1. SB: - the numerical Seltzer-Berger DCS for the emitted photon energy.
//        - used between 1 keV - 1 GeV primary e-/e+ kinetic energies.
//        NOTE: the core part i.e. sampling the emitted photon energy is different than
//          that in the G4SeltzerBergerModel. I implemented here my rejection free,
//          memory effcicient (tables only per Z and not per mat-cuts) sampling.
//          Rejection is used only to account dielectric supression and e+ correction.
// 2. RB: - the Bethe-Heitler DCS with modifications such as screening and Coulomb
//          corrections, emission in the field of the atomic electrons and LPM suppression.
//          Used between 1 GeV - 100 TeV primary e-/e+ kinetic energies.
void PerformElectronBrem(G4HepEmTLData* tlData, struct G4HepEmData* hepEmData, bool iselectron, bool isSBmodel);


// Sampling of the energy transferred to the emitted photon using the numerical
// Seltzer-Berger DCS.
G4HepEmHostDevice
double SampleETransferBremSB(struct G4HepEmData* hepEmData, double thePrimEkin, double theLogEkin,
                             int theIMCIndx, G4HepEmRandomEngine* rnge, bool iselectron);

// Sampling of the energy transferred to the emitted photon using the Bethe-Heitler
// DCS.
G4HepEmHostDevice
double SampleETransferBremRB(struct G4HepEmData* hepEmData, double thePrimEkin, double theLogEkin,
                             int theIMCIndx, G4HepEmRandomEngine* rnge, bool iselectron);


// Target atom selector for the above bremsstrahlung intercations in case of
// materials composed from multiple elements.
G4HepEmHostDevice
int SelectTargetAtomBrem(const struct G4HepEmElectronData* elData, const int imc, const double ekin,
                         const double lekin, const double urndn, const bool isbremSB);


G4HepEmHostDevice
void SampleDirectionsBrem(const double thePrimEkin, const double theSecGammaEkin, double* theSecGammaDir,
                          double* thePrimElecDir, G4HepEmRandomEngine* rnge);


// Simple linear search (with step of 3!) used in the photon energy sampling part
// of the SB (Seltzer-Berger) brem model.
G4HepEmHostDevice
int LinSearch(const double* vect, const int size, const double val);


#endif // G4HepEmElectronInteractionBrem_HH
