
#ifndef Declaration_HH
#define Declaration_HH

// G4 include (for types)
#include "globals.hh"

class G4MaterialCutsCouple;

struct G4HepEmData;
struct G4HepEmElectronData;


void G4SBTest     (const G4MaterialCutsCouple* g4MatCut, G4double ekin, G4double numSamples, G4int numHistBins, G4bool isSBmodel, G4bool iselectron=true);
void G4HepEmSBTest(const G4MaterialCutsCouple* g4MatCut, G4double ekin, G4double numSamples, G4int numHistBins, G4bool isSBmodel, G4bool iselectron=true);

#ifdef G4HepEm_CUDA_BUILD

#include <device_launch_parameters.h>

  bool TestSBTableData(const struct G4HepEmData* hepEmData);

  __global__
  void TestSBTableDataKernel(struct G4HepEmSBTableData* theTableData_d, int* theIData1_d, int* theIData2_d, int* theIData3_d, int* theIData4_d,
                             double* theDData1_d, double* theDData2_d, double* theDData3_d, double* theDData4_d, double* theDData5_d);

#endif // G4HepEm_CUDA_BUILD


#endif // Declaration_HH
