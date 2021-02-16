
#ifndef G4SETUP_HH
#define G4SETUP_HH

// G4 include (for types)
#include "globals.hh"

class G4MaterialCutsCouple;


/**
 * @file    G4SetUp.hh
 * @author  M. Novak
 * @date    2021
 *
 * Simple utility functions to construct a fake Geant4 detector geometry.
 *
 * The detector can be constructed either with a single, specific target material
 * or including all pre-defined NIST materials. The secondary production threshold
 * can also be spefified.
 */


// builds a fake Geant4 geometry including all G4-NIST materials
void FakeG4Setup ( G4double prodCutInLength,  G4int verbose=1 );

// builds a fake Geant4 geometry with a single G4-NIST material
const G4MaterialCutsCouple*
FakeG4Setup ( G4double prodCutInLength, const G4String& nistMatName, G4int verbose=1);






#endif // G4SETUP_HH
