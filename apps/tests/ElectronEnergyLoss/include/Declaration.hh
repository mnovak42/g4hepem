
#ifndef Declaration_HH
#define Declaration_HH

// G4 include (for types)
#include "globals.hh"


struct G4HepEmData;
struct G4HepEmElectronDataOnDevice;


// builds a fake Geant4 geometry just to be able to produce material-cuts couple
void FakeG4Setup ( G4double prodCutInLength,  G4int verbose=1 );


// checks the EnergyLoss related parts of the G4HepEmElectronData (host/device)
bool TestElossData ( const struct G4HepEmData* hepEmData, bool iselectron=true );


#ifdef G4HepEm_CUDA_BUILD 

#include <device_launch_parameters.h>

  // kernel to evaluate the Range and dE/dx data (they stored in the same way)
  template <bool TisRange>
  __global__
  void TestElossDataRangeDEDXKernel ( struct G4HepEmElectronDataOnDevice* theElectronData_d, 
                                      int* tsInImc_d, double* tsInEkin_d, double* tsInLogEkin_d,
                                      double* tsOutRes_d, int numTestCases );

  // kernels to evaluate the inverse Range data:
  //  - the lower index of the discrete range value bin in which the give test
  //    case is located, needs to be determined that requires a search
  //  - the devie side fuction does this (more efficient solutions can be given
  //    is more information is available on the access pattern)
  __device__ 
  int   TestElossDataInvRangeFindBin ( double* theRangeArray_d, int itsSize, double theRangeVal );

  __global__
  void TestElossDataInvRangeKernel ( struct G4HepEmElectronDataOnDevice* theElectronData_d, 
                                     int* tsInImc_d, double* tsInRange_d,
                                     double* tsOutRes_d, int numTestCases );
                                  
                                  
  // Evaluates all test cases on the device for computing the range, dE/dx and inverse 
  // range values on the device for all test cases.
  void TestElossDataOnDevice ( const struct G4HepEmData* hepEmData, 
                               int* tsInImc_h, double* tsInEkin_h, double* tsInLogEkin_h,
                               double* tsOutResRange_h, double* tsOutResDEDX_h, double* tsOutResInvRange_h,  
                               int numTestCases, bool iselectron );





#endif // G4HepEm_CUDA_BUILD 


#endif // Declaration_HH


