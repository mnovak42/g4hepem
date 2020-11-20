
#ifndef G4HepEmElectronInteractionIoni_HH
#define G4HepEmElectronInteractionIoni_HH

class  G4HepEmTLData;
struct G4HepEmData;


// Ionisation interaction for e-/e+ described by the Moller/Bhabha model.
// Used between 100 eV - 100 TeV primary e-/e+ kinetic energies.
void PerformElectronIoni(G4HepEmTLData* tlData, struct G4HepEmData* hepEmData, bool iselectron);

// Sampling of the energy transferred to the secondary electron in case of e- 
// primary i.e. in case of Moller interaction.
double SampleETransferMoller(const double elCut, const double primEkin, G4HepEmTLData* tlData);
// Sampling of the energy transferred to the secondary electron in case of e+ 
// primary i.e. in case of Bhabha interaction.
double SampleETransferBhabha(const double elCut, const double primEkin, G4HepEmTLData* tlData);

#endif // G4HepEmElectronInteractionIoni_HH