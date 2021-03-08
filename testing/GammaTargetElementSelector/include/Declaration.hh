
#ifndef Declaration_HH
#define Declaration_HH

struct G4HepEmData;
struct G4HepEmParameters;
struct G4HepEmGammaData;


// checks the Target Element Selector data (gamma: conversion) of the G4HepEmGammaData (host/device)
bool TestGammaElemSelectorData ( const struct G4HepEmData* hepEmData );


#ifdef G4HepEm_CUDA_BUILD

  // Evaluates all test cases on the device by sampling the index of the target element (of the
  // target material) on which the interaction takes place, on the device for all test cases.
  void TestGammaElemSelectorDataOnDevice ( const struct G4HepEmData* hepEmData, int* tsInImat_h,
                                      double* tsInEkin_h, double* tsInLogEkin_h, double* tsInRngVals_h,
                                      int* tsOutRes_h, int numTestCases );

#endif // G4HepEm_CUDA_BUILD


#endif // Declaration_HH
