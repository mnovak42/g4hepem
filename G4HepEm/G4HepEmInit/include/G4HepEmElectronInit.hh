
// Initialisation of all e-/e+ related data (e-loss tables, macroscopic cross 
// sections and target element selectors for each model) and interaction models.

#ifndef G4HepEmElementInit_HH
#define G4HepEmElementInit_HH

struct G4HepEmData;
struct G4HepEmParameters;

void InitElectronData(struct G4HepEmData* hepEmData, struct G4HepEmParameters* hepEmPars, bool iselectron);



#endif // G4HepEmElementInit_HH