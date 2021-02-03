
#ifndef Declaration_HH
#define Declaration_HH

// G4 include (for types)
#include "globals.hh"

class G4MaterialCutsCouple;


struct G4HepEmData;
struct G4HepEmElectronData;


// builds a fake Geant4 geometry including all G4-NIST materials
void FakeG4Setup ( G4double prodCutInLength,  G4int verbose=1 );

// builds a fake Geant4 geometry with a single G4-NIST material
const G4MaterialCutsCouple*
FakeG4Setup ( G4double prodCutInLength, const G4String& nistMatName, G4int verbose=1);


void G4SBTest     (const G4MaterialCutsCouple* g4MatCut, G4double ekin, G4double numSamples, G4int numHistBins, G4bool isSBmodel, G4bool iselectron=true);
void G4HepEmSBTest(const G4MaterialCutsCouple* g4MatCut, G4double ekin, G4double numSamples, G4int numHistBins, G4bool isSBmodel, G4bool iselectron=true);


#endif // Declaration_HH
