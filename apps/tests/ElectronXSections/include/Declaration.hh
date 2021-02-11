
#ifndef Declaration_HH
#define Declaration_HH

struct G4HepEmData;
struct G4HepEmElectronData;

// checks the Cross section  related parts of the G4HepEmElectronData (host/device)
bool TestXSectionData ( const struct G4HepEmData* hepEmData, bool iselectron=true );


#ifdef G4HepEm_CUDA_BUILD

  // Evaluates all test cases on the device for computing the restricted macroscopic
  // cross section values for ionisation and bremsstrahlung on the device for all test cases.
  void TestResMacXSecDataOnDevice ( const struct G4HepEmData* hepEmData, int* tsInImc_h,
                                    double* tsInEkinIoni_h, double* tsInLogEkinIoni_h,
                                    double* tsInEkinBrem_h, double* tsInLogEkinBrem_h,
                                    double* tsOutResMXIoni_h, double* tsOutResMXBrem_h,
                                    int numTestCases, bool iselectron );

#endif // G4HepEm_CUDA_BUILD


#endif // Declaration_HH
