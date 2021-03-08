
#ifndef Declaration_HH
#define Declaration_HH

struct G4HepEmData;
struct G4HepEmGammaData;

// checks the Cross section  related parts of the G4HepEmGammaData (host/device)
bool TestGammaXSectionData ( const struct G4HepEmData* hepEmData );


#ifdef G4HepEm_CUDA_BUILD

  // Evaluates all test cases on the device for computing the macroscopic
  // cross section values for conversion and Compton on the device for all test cases.
  void TestMacXSecDataOnDevice ( const struct G4HepEmData* hepEmData, int* tsInImat_h,
                                 double* tsInEkinConv_h, double* tsInLogEkinConv_h,
                                 double* tsInEkinComp_h, double* tsInLogEkinComp_h,
                                 double* tsOutMXConv_h,  double* tsOutMXComp_h,
                                 int numTestCases );

#endif // G4HepEm_CUDA_BUILD


#endif // Declaration_HH
