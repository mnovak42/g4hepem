
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


void G4SBTest     (const G4MaterialCutsCouple* g4MatCut, G4double ekin, G4double numSamples, G4int numHistBins, G4bool iselectron=true);
void G4HepEmSBTest(const G4MaterialCutsCouple* g4MatCut, G4double ekin, G4double numSamples, G4int numHistBins, G4bool iselectron=true);


#ifdef G4HepEm_CUDA_BUILD

#include <device_launch_parameters.h>

  // Samples the emitted photon energy on the device.
  void G4HepEmSBTestOnDevice ( const struct G4HepEmData* hepEmData, int* tsInImc_h,
                                      double* tsInEkin_h, double* tsInLogEkin_h, double* tsInRngVals_h,
                                      int* tsOutRes_h, int numTestCases, int indxModel, bool iselectron );

  __global__
  void G4HepEmSBTestKernel ( const struct G4HepEmElectronData* theElectronData_d,
                                        const struct G4HepEmMatCutData* theMatCutData_d,
                                        const struct G4HepEmMaterialData* theMaterialData_d,
                                        int* tsInImc_d, double* tsInEkin_d, double* tsInLogEkin_d, double* tsInRngVals_d,
                                        int* tsOutRes_d, int numTestCases );


#endif // G4HepEm_CUDA_BUILD


#endif // Declaration_HH
