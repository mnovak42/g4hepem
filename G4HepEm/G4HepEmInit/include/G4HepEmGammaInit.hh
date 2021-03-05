
// Initialisation of all gamma related data (macroscopic cross sections
// and target element selectors for each model) and interaction models.

// NOTE: only Conversion and Compton is active at the moment

#ifndef G4HepEmGammaInit_HH
#define G4HepEmGammaInit_HH

struct G4HepEmData;
struct G4HepEmParameters;

void InitGammaData(struct G4HepEmData* hepEmData, struct G4HepEmParameters* hepEmPars);



#endif // G4HepEmGammaInit_HH
