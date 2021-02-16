
#include "Declaration.hh"


// G4HepEm includes
#include "G4HepEmData.hh"
#include "G4HepEmMatCutData.hh"
#include "G4HepEmElectronData.hh"

// don't worry it's just for testing
#define private public
#include "G4HepEmElectronManager.hh"
#undef private

#include <cmath>
#include <random>
#include <iostream>
#include <iomanip>


bool TestXSectionData ( const struct G4HepEmData* hepEmData, bool iselectron ) {
  bool isPassed     = true;
  // number of mat-cut and kinetic energy pairs go generate and test
  int  numTestCases = 32768;
  // set up an rng to get mc-indices on [0,numMCData)
  std::random_device rd;
  std::mt19937 gen(rd());
  gen.seed(0); // fix seed
  std::uniform_real_distribution<> dis(0, 1.0);
  // get ptr to the G4HepEmElectronData and G4HepEmMatCutData structures
  const G4HepEmElectronData* theElectronData = iselectron ? hepEmData->fTheElectronData : hepEmData->fThePositronData;
  const int numELossData = theElectronData->fELossEnergyGridSize;
  const G4HepEmMatCutData*   theMatCutData   = hepEmData->fTheMatCutData;
  const int numMCData    = theMatCutData->fNumMatCutData;
  // allocate memory (host) to store the generated test cases:
  //  - the numTestCases, material-cut index and kinetic energy combinations
  // and the results:
  //  - the numTestCases, restricted macroscopic cross sction for ionisation, bremsstrahlung
  //    evaluated at test cases.
  int*    tsInImc         = new int[numTestCases];
  double* tsInEkinIoni    = new double[numTestCases];
  double* tsInLogEkinIoni = new double[numTestCases];
  double* tsInEkinBrem    = new double[numTestCases];
  double* tsInLogEkinBrem = new double[numTestCases];
  double* tsOutResMXIoni  = new double[numTestCases];
  double* tsOutResMXBrem  = new double[numTestCases];
  // the maximum (+2%) primary particle kinetic energy that is covered by the simulation (100 TeV by default)
  const double    maxEKin = 1.02*theElectronData->fELossEnergyGrid[numELossData-1];
  for (int i=0; i<numTestCases; ++i) {
    int imc            = (int)(dis(gen)*numMCData);
    tsInImc[i]         = imc;
    // == Ionisation:
    // get the min/max of the possible prirmary e-/e+ kinetic energies at which
    // the restricted interacton can happen in this material-cuts (use +- 2% out of range)
    double secElCutE   = theMatCutData->fMatCutData[imc].fSecElProdCutE;
    double minEKin     = iselectron ? 0.98*2.0*secElCutE : 0.98*secElCutE;
    // generate a unifomly random kinetic energy point in the allowed (+- 2%) primary
    // particle kinetic energy range on logarithmic scale
    double lMinEkin    = std::log(minEKin);
    double lEkinDelta  = std::log(maxEKin/minEKin);
    tsInLogEkinIoni[i] = dis(gen)*lEkinDelta+lMinEkin;
    tsInEkinIoni[i]    = std::exp(tsInLogEkinIoni[i]);
    // == Bremsstrahlung: (the same with different limits)
    minEKin            = 0.98*theMatCutData->fMatCutData[imc].fSecGamProdCutE;
    lMinEkin           = std::log(minEKin);
    lEkinDelta         = std::log(maxEKin/minEKin);
    tsInLogEkinBrem[i] = dis(gen)*lEkinDelta+lMinEkin;
    tsInEkinBrem[i]    = std::exp(tsInLogEkinBrem[i]);
  }
  //
  // Use a G4HepEmElectronManager object to evaluate the restricted macroscopic
  // cross sections for ionisation and bremsstrahlung for the test cases.
  G4HepEmElectronManager theElectronMgr;
  for (int i=0; i<numTestCases; ++i) {
    tsOutResMXIoni[i] = theElectronMgr.GetRestMacXSec (theElectronData, tsInImc[i], tsInEkinIoni[i], tsInLogEkinIoni[i], true);
    tsOutResMXBrem[i] = theElectronMgr.GetRestMacXSec (theElectronData, tsInImc[i], tsInEkinBrem[i], tsInLogEkinBrem[i], false);
  }


#ifdef G4HepEm_CUDA_BUILD
  //
  // Perform the test case evaluations on the device
  double* tsOutResOnDeviceMXIoni = new double[numTestCases];
  double* tsOutResOnDeviceMXBrem = new double[numTestCases];
  TestResMacXSecDataOnDevice (hepEmData, tsInImc, tsInEkinIoni, tsInLogEkinIoni, tsInEkinBrem, tsInLogEkinBrem, tsOutResOnDeviceMXIoni, tsOutResOnDeviceMXBrem, numTestCases, iselectron);
  for (int i=0; i<numTestCases; ++i) {
//    std::cout << tsInEkinIoni[i] << " "<<tsOutResMXIoni[i] << " " << tsOutResOnDeviceMXIoni[i] << " " <<tsInEkinBrem[i] << " " << tsOutResMXBrem[i] << " " << tsOutResOnDeviceMXBrem[i] << std::endl;
    if ( std::abs( 1.0 - tsOutResMXIoni[i]/tsOutResOnDeviceMXIoni[i] ) > 1.0E-14 ) {
      isPassed = false;
      std::cerr << "\n*** ERROR:\nRestricted Macroscopic Cross Section data: G4HepEm Host vs Device (Ioni) mismatch: " << std::setprecision(16) << tsOutResMXIoni[i] << " != " << tsOutResOnDeviceMXIoni[i] << " ( i = " << i << " imc  = " << tsInImc[i] << " ekin =  " << tsInEkinIoni[i] << ") " << std::endl;
      break;
    }
    if ( std::abs( 1.0 - tsOutResMXBrem[i]/tsOutResOnDeviceMXBrem[i] ) > 1.0E-14 ) {
      isPassed = false;
      std::cerr << "\n*** ERROR:\nRestricted Macroscopic Cross Section data: G4HepEm Host vs Device (Brem) mismatch: " <<  std::setprecision(16) << tsOutResMXBrem[i] << " != " << tsOutResOnDeviceMXBrem[i] << " ( i = " << i << " imc  = " << tsInImc[i] << " ekin =  " << tsInEkinBrem[i] << ") " << std::endl;
      break;
    }
  }
  //
  delete [] tsOutResOnDeviceMXIoni;
  delete [] tsOutResOnDeviceMXBrem;
#endif // G4HepEm_CUDA_BUILD

  //
  // delete allocatd memeory
  delete [] tsInImc;
  delete [] tsInEkinIoni;
  delete [] tsInLogEkinIoni;
  delete [] tsInEkinBrem;
  delete [] tsInLogEkinBrem;
  delete [] tsOutResMXIoni;
  delete [] tsOutResMXBrem;

  return isPassed;
}
