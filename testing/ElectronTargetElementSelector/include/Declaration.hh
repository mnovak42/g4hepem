
#ifndef Declaration_HH
#define Declaration_HH

struct G4HepEmData;
struct G4HepEmParameters;
struct G4HepEmElectronData;


// checks the Target Element Selector data (e-/e+: ioni and brem models) of the G4HepEmElectronData (host/device)
bool TestElemSelectorData ( const struct G4HepEmData* hepEmData, const struct G4HepEmParameters* hepEmParams, bool iselectron=true );


#ifdef G4HepEm_CUDA_BUILD

  // Evaluates all test cases on the device by sampling the index of the target element (of the
  // target material-cuts couple) on which the interaction takes place, on the device for all
  // test cases.
  void TestElemSelectorDataOnDevice ( const struct G4HepEmData* hepEmData, int* tsInImc_h,
                                      double* tsInEkin_h, double* tsInLogEkin_h, double* tsInRngVals_h,
                                      int* tsOutRes_h, int numTestCases, int indxModel, bool iselectron );

#endif // G4HepEm_CUDA_BUILD


#endif // Declaration_HH
