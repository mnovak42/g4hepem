
#ifndef Declaration_HH
#define Declaration_HH

// G4 include (for types)
#include "globals.hh"


struct G4HepEmData;
struct G4HepEmParameters;
struct G4HepEmElectronDataOnDevice;


// builds a fake Geant4 geometry just to be able to produce material-cuts couple
void FakeG4Setup ( G4double prodCutInLength,  G4int verbose=1 );


// checks the Target Element Selector data (e-/e+: ioni and brem models) of the G4HepEmElectronData (host/device)
bool TestElemSelectorData ( const struct G4HepEmData* hepEmData, const struct G4HepEmParameters* hepEmParams, bool iselectron=true );


#ifdef G4HepEm_CUDA_BUILD 

#include <device_launch_parameters.h>

  // Kernel to evaluate/sample target element on which the interaction takes place 
  // for the test cases. The required data are stored in the ElemSelector data part of the 
  // G4HepEmElectronData structrue.
  //
  template <int TmodelIndx>
  __global__
  void TestElemSelectorDataKernel ( const struct G4HepEmElectronDataOnDevice* theElectronData_d, 
                                    int* tsInImc_d, double* tsInEkin_d, double* tsInLogEkin_d, double tsInRngVals_d,
                                    int* tsOutRes_d, int numTestCases );
 
  // Evaluates all test cases on the device by sampling the index of the target element (of the 
  // target material-cuts couple) on which the interaction takes place, on the device for all 
  // test cases.
  void TestElemSelectorDataOnDevice ( const struct G4HepEmData* hepEmData, int* tsInImc_h, 
                                      double* tsInEkin_h, double* tsInLogEkin_h, double* tsInRngVals_h,
                                      int* tsOutRes_h, int numTestCases, int indxModel, bool iselectron );

#endif // G4HepEm_CUDA_BUILD 


#endif // Declaration_HH


