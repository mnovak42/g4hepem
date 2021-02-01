
#ifndef Declaration_HH
#define Declaration_HH

// G4 include (for types)
#include "globals.hh"


struct G4HepEmData;
struct G4HepEmElectronData;


// builds a fake Geant4 geometry just to be able to produce material-cuts couple
void FakeG4Setup ( G4double prodCutInLength,  G4int verbose=1 );


// checks the Cross section  related parts of the G4HepEmElectronData (host/device)
bool TestXSectionData ( const struct G4HepEmData* hepEmData, bool iselectron=true );


#ifdef G4HepEm_CUDA_BUILD

#include <device_launch_parameters.h>


  // Evaluates all test cases on the device for computing the restricted macroscopic
  // cross section values for ionisation and bremsstrahlung on the device for all test cases.
  void TestResMacXSecDataOnDevice ( const struct G4HepEmData* hepEmData, int* tsInImc_h,
                                    double* tsInEkinIoni_h, double* tsInLogEkinIoni_h,
                                    double* tsInEkinBrem_h, double* tsInLogEkinBrem_h,
                                    double* tsOutResMXIoni_h, double* tsOutResMXBrem_h,
                                    int numTestCases, bool iselectron );

  __global__
  void TestResMacXSecDataKernel ( const struct G4HepEmElectronData* theElectronData_d,
                                  int* tsInImc_d, double* tsInEkin_d, double* tsInLogEkin_d,
                                  double* tsOutRes_d, bool isIoni, int numTestCases );

#endif // G4HepEm_CUDA_BUILD


#endif // Declaration_HH
