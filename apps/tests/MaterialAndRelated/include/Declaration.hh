
#ifndef Declaration_HH
#define Declaration_HH

// G4 include (for types)
#include "globals.hh"


// builds a fake Geant4 geometry just to be able to produce material-cuts couple
void FakeG4Setup ( G4double prodCutInLength,  G4int verbose=1 );


// checks G4HepEmElemData (host) by comparing to those in Geant4
bool TestElementData   ( const struct G4HepEmData* hepEmData );
bool TestMaterialData  ( const struct G4HepEmData* hepEmData );
bool TestMatCutData    ( const struct G4HepEmData* hepEmData );


#ifdef G4HepEm_CUDA_BUILD 

#include <device_launch_parameters.h>

  __global__
  void TestElementDataKernel    ( struct G4HepEmElementData* elemData_d, int* elemIndices_d, 
                                  double* resZet_d, double* resZet13_d, int numTestCases);
  bool TestElementDataOnDevice  ( const struct G4HepEmData* hepEmData );

  __global__
  void TestMaterialDataKernel   ( struct G4HepEmMaterialData* matData_d, int* matIndices_d, int* indxStarts_d, 
                                  double* resCompADens_d, int* resCompElems_d,  int* resNumElems_d, 
                                  double* resMassDens_d, double* resElecDens_d, double* resRadLen_d, int numTestCases );
  bool TestMaterialDataOnDevice ( const struct G4HepEmData* hepEmData );

  __global__
  void TestMatCutDataKernel     ( struct G4HepEmMatCutData* mcData_d, int* mcIndices_d,
                                  double* resSecElCut_d, double* resSecGamCut_d, int* resMatIndx_d, int numTestCases );
  bool TestMatCutDataOnDevice   ( const struct G4HepEmData* hepEmData );

#endif // G4HepEm_CUDA_BUILD 


#endif // Declaration_HH


