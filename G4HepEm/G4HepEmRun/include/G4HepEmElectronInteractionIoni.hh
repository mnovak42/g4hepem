
#ifndef G4HepEmElectronInteractionIoni_HH
#define G4HepEmElectronInteractionIoni_HH

#include "G4HepEmMacros.hh"

class  G4HepEmTLData;
class  G4HepEmRandomEngine;
struct G4HepEmData;


// Ionisation interaction for e-/e+ described by the Moller/Bhabha model.
// Used between 100 eV - 100 TeV primary e-/e+ kinetic energies.
void PerformElectronIoni(G4HepEmTLData* tlData, struct G4HepEmData* hepEmData, bool iselectron);

// Sampling of the energy transferred to the secondary electron in case of e-
// primary i.e. in case of Moller interaction.
G4HepEmHostDevice
double SampleETransferMoller(const double elCut, const double primEkin, G4HepEmRandomEngine* rnge);
// Sampling of the energy transferred to the secondary electron in case of e+
// primary i.e. in case of Bhabha interaction.
G4HepEmHostDevice
double SampleETransferBhabha(const double elCut, const double primEkin, G4HepEmRandomEngine* rnge);

G4HepEmHostDevice
void SampleDirectionsIoni(const double thePrimEkin, const double deltaEkin, double* theSecElecDir, double* thePrimElecDir, G4HepEmRandomEngine* rnge);

#endif // G4HepEmElectronInteractionIoni_HH
