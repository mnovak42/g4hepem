
#ifndef Declaration_HH
#define Declaration_HH

// G4 include (for types)
#include "globals.hh"


struct G4HepEmData;
struct G4HepEmElectronDataOnDevice;


// builds a fake Geant4 geometry just to be able to produce material-cuts couple
void FakeG4Setup ( G4double prodCutInLength,  G4int verbose=1 );


// checks the Cross section  related parts of the G4HepEmElectronData (host/device)
bool TestXSectionData ( const struct G4HepEmData* hepEmData, bool iselectron=true );


#ifdef G4HepEm_CUDA_BUILD 

#include <device_launch_parameters.h>

  // Kernel to evaluate the Restricted macroscopic cross section values for the 
  // test cases. The required data are stored in the ResMacXSec data part of the 
  // G4HepEmElectronData structrue.
  //
  // Note: both specialisations (needed to be called from the host) are done in 
  //  the .cu file in the TestResMacXSecDataOnDevice function.
  template <bool TIsIoni>
  __global__
  void TestResMacXSecDataKernel ( struct G4HepEmElectronDataOnDevice* theElectronData_d, 
                                  int* tsInImc_d, double* tsInEkin_d, double* tsInLogEkin_d,
                                  double* tsOutRes_d, int numTestCases );
 
  // Evaluates all test cases on the device for computing the restricted macroscopic
  // cross section values for ionisation and bremsstrahlung on the device for all test cases.
  void TestResMacXSecDataOnDevice ( const struct G4HepEmData* hepEmData, int* tsInImc_h, 
                                    double* tsInEkinIoni_h, double* tsInLogEkinIoni_h,
                                    double* tsInEkinBrem_h, double* tsInLogEkinBrem_h,
                                    double* tsOutResMXIoni_h, double* tsOutResMXBrem_h, 
                                    int numTestCases );

#endif // G4HepEm_CUDA_BUILD 


#endif // Declaration_HH


