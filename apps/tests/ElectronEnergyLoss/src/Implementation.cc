
#include "Declaration.hh"

// G4HepEm includes
#include "G4HepEmData.hh"
#include "G4HepEmMatCutData.hh"
#include "G4HepEmElectronData.hh"

// don't worry it's just for testing
#define private public
#include "G4HepEmElectronManager.hh"

#include <vector>
#include <cmath>
#include <random>
#include <iostream>


bool TestElossData ( const struct G4HepEmData* hepEmData, bool iselectron ) {
  bool isPassed     = true;
  // number of mat-cut and kinetic energy pairs go generate and test
  int  numTestCases = 32768;
  // number of mat-cut data i.e. G4HepEm mat-cut indices are in [0,numMCData)
  int  numMCData    = hepEmData->fTheMatCutData->fNumMatCutData;
  // set up an rng to get mc-indices on [0,numMCData)
  std::random_device rd;
  std::mt19937 gen(rd());
  gen.seed(0); // fix seed
  std::uniform_real_distribution<> dis(0, 1.0);
  // get ptr to the G4HepEmElectronData structure
  const G4HepEmElectronData* theElectronData = iselectron ? hepEmData->fTheElectronData : hepEmData->fThePositronData;
  // for the generation of test particle kinetic energy values:
  // - get the min/max values of the energy loss (related data) kinetic energy grid
  // - also the number of discrete kinetic energy grid points (used later)
  // - test particle kinetic energies will be generated uniformly random, on log
  //   kinetic energy scale, between +- 5 percent of the limits (in order to test
  //   below above grid limits cases as well)
  const int     numELossData = theElectronData->fELossEnergyGridSize;
  const double  minELoss     = 0.95*theElectronData->fELossEnergyGrid[0];
  const double  maxELoss     = 1.05*theElectronData->fELossEnergyGrid[numELossData-1];
  // allocate memory (host) to store the generated test cases:
  //  - the numTestCases, material-cut index and kinetic energy combinations
  // and the results:
  //  - the numTestCases, restricted dEdx, range and inverse-range values for the
  //    test cases.
  int*    tsInImc           = new int[numTestCases];
  double* tsInEkin          = new double[numTestCases];
  double* tsInLogEkin       = new double[numTestCases];
  double* tsOutResRange     = new double[numTestCases];
  double* tsOutResDEDX      = new double[numTestCases];
  double* tsOutResInvRange  = new double[numTestCases];
  // generate the test cases: mat-cut indices and kinetic energy combinations
  const double lMinELoss   = std::log(minELoss);
  const double lELossDelta = std::log(maxELoss/minELoss);
  for (int i=0; i<numTestCases; ++i) {
    tsInImc[i]     = (int)(dis(gen)*numMCData);
    tsInLogEkin[i] = dis(gen)*lELossDelta+lMinELoss;
    tsInEkin[i]    = std::exp(tsInLogEkin[i]);
  }
  //
  // Use a G4HepEmElectronManager object to evaluate the range, dedx and inverse-range
  // values for the test cases.
  G4HepEmElectronManager theElectronMgr;
  for (int i=0; i<numTestCases; ++i) {
    tsOutResRange[i]    = theElectronMgr.GetRestRange(theElectronData, tsInImc[i], tsInEkin[i], tsInLogEkin[i]);
    tsOutResDEDX[i]     = theElectronMgr.GetRestDEDX (theElectronData, tsInImc[i], tsInEkin[i], tsInLogEkin[i]);
    tsOutResInvRange[i] = theElectronMgr.GetInvRange (theElectronData, tsInImc[i], tsOutResRange[i]);
  }


#ifdef G4HepEm_CUDA_BUILD
  //
  // Perform the test case evaluations on the device
  double* tsOutResOnDeviceRange    = new double[numTestCases];
  double* tsOutResOnDeviceDEDX     = new double[numTestCases];
  double* tsOutResOnDeviceInvRange = new double[numTestCases];
  TestElossDataOnDevice (hepEmData, tsInImc, tsInEkin, tsInLogEkin, tsOutResOnDeviceRange, tsOutResOnDeviceDEDX, tsOutResOnDeviceInvRange, numTestCases, iselectron);
  for (int i=0; i<numTestCases; ++i) {
    if ( std::abs( 1.0 - tsOutResRange[i]/tsOutResOnDeviceRange[i] ) > 1.0E-14 ) {
      isPassed = false;
      std::cerr << "\n*** ERROR:\nEnergyLoss data: G4HepEm Host vs Device RANGE mismatch: " << tsOutResRange[i] << " != " << tsOutResOnDeviceRange[i] << " ( i = " << i << " imc  = " << tsInImc[i] << " ekin =  " << tsInEkin[i] << ") " << std::endl;
      break;
    }
    if ( std::abs( 1.0 - tsOutResDEDX[i]/tsOutResOnDeviceDEDX[i] ) > 1.0E-14 ) {
      isPassed = false;
      std::cerr << "\n*** ERROR:\nEnergyLoss data: G4HepEm Host vs Device dE/dx mismatch: "  << tsOutResDEDX[i] << " != " << tsOutResOnDeviceDEDX[i] << " ( i = " << i << " imc  = " << tsInImc[i] << " ekin =  " << tsInEkin[i] << ") " << std::endl;
      break;
    }
    if ( std::abs( 1.0 - tsOutResInvRange[i]/tsOutResOnDeviceInvRange[i] ) > 1.0E-14 ) {
      isPassed = false;
      std::cerr << "\n*** ERROR:\nEnergyLoss data: G4HepEm Host vs Device Inverse-RANGE mismatch: "  << tsOutResInvRange[i] << " != " << tsOutResOnDeviceInvRange[i] << " ( i = " << i << " imc  = " << tsInImc[i] << " ekin =  " << tsInEkin[i] << " range =  " << tsOutResRange[i]<< ") " << std::endl;
      break;
    }
  }
  //
  delete [] tsOutResOnDeviceRange;
  delete [] tsOutResOnDeviceDEDX;
  delete [] tsOutResOnDeviceInvRange;
#endif // G4HepEm_CUDA_BUILD


  //
  // delete allocatd memeory
  delete [] tsInImc;
  delete [] tsInEkin;
  delete [] tsInLogEkin;
  delete [] tsOutResRange;
  delete [] tsOutResDEDX;
  delete [] tsOutResInvRange;

  return isPassed;
}
