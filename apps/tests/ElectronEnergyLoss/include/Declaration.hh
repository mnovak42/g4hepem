
#ifndef Declaration_HH
#define Declaration_HH

struct G4HepEmData;
struct G4HepEmElectronData;

// checks the EnergyLoss related parts of the G4HepEmElectronData (host/device)
bool TestElossData ( const struct G4HepEmData* hepEmData, bool iselectron=true );


#ifdef G4HepEm_CUDA_BUILD

  // Evaluates all test cases on the device for computing the range, dE/dx and inverse
  // range values on the device for all test cases.
  void TestElossDataOnDevice ( const struct G4HepEmData* hepEmData,
                               int* tsInImc_h, double* tsInEkin_h, double* tsInLogEkin_h,
                               double* tsOutResRange_h, double* tsOutResDEDX_h, double* tsOutResInvRange_h,
                               int numTestCases, bool iselectron );

#endif // G4HepEm_CUDA_BUILD


#endif // Declaration_HH
