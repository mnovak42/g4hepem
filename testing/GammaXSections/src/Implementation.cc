
#include "Declaration.hh"


// G4HepEm includes
#include "G4HepEmData.hh"
#include "G4HepEmMaterialData.hh"
#include "G4HepEmGammaData.hh"

// don't worry it's just for testing
#define private public
#include "G4HepEmGammaManager.hh"
#undef private

#include <cmath>
#include <random>
#include <iostream>
#include <iomanip>


bool TestGammaXSectionData ( const struct G4HepEmData* hepEmData ) {
  bool isPassed     = true;
  // number of mat and kinetic energy pairs to generate and test
  int  numTestCases = 32768;
  // set up an rng to get mat-indices on [0,numMat)
  std::random_device rd;
  std::mt19937 gen(rd());
  gen.seed(0); // fix seed
  std::uniform_real_distribution<> dis(0, 1.0);
  // get ptr to the G4HepEmGammaData and G4HepEmMaterialData structures
  const G4HepEmGammaData* theGammaData  = hepEmData->fTheGammaData;
  const G4HepEmMaterialData* theMatData = hepEmData->fTheMaterialData;

  const int numConvData = theGammaData->fConvEnergyGridSize;
  const int numCompData = theGammaData->fCompEnergyGridSize;
  const int numMatData  = theMatData->fNumMaterialData;
  // allocate memory (host) to store the generated test cases:
  //  - the numTestCases, material index and kinetic energy combinations
  // and the results:
  //  - the numTestCases, restricted macroscopic cross sction for conversion,
  //    Compton scattering evaluated at the test cases.
  int*    tsInImat        = new int[numTestCases];
  double* tsInEkinConv    = new double[numTestCases];
  double* tsInLogEkinConv = new double[numTestCases];
  double* tsInEkinComp    = new double[numTestCases];
  double* tsInLogEkinComp = new double[numTestCases];
  double* tsOutMXConv     = new double[numTestCases];
  double* tsOutMXComp     = new double[numTestCases];
  // the maximum (+2%) primary particle kinetic energy that is covered by the
  // simulation (100 TeV by default). alos use -2% for the low energy limit.
  const double     maxLEKin = std::log(1.02*theGammaData->fConvEnergyGrid[numConvData-1]);
  const double minLEKinConv = std::log(theGammaData->fConvEnergyGrid[0]*0.98);
  const double minLEKinComp = std::log(theGammaData->fCompEnergyGrid[0]*0.98);
  for (int i=0; i<numTestCases; ++i) {
    int imat           = (int)(dis(gen)*numMatData);
    tsInImat[i]        = imat;
    // -- conversion
    double lMinEkin    = minLEKinConv;
    double lEkinDelta  = maxLEKin - minLEKinConv;
    tsInLogEkinConv[i] = dis(gen)*lEkinDelta+minLEKinConv;
    tsInEkinConv[i]    = std::exp(tsInLogEkinConv[i]);
    // -- Compton
    lMinEkin           = minLEKinComp;
    lEkinDelta         = maxLEKin - minLEKinComp;
    tsInLogEkinComp[i] = dis(gen)*lEkinDelta+minLEKinComp;
    tsInEkinComp[i]    = std::exp(tsInLogEkinComp[i]);
  }
  //
  // Use a G4HepEmGammaManager object to evaluate the macroscopic cross sections
  // for conversion inot e-e+ pairs and Compton scattering.
  G4HepEmGammaManager theGammaMgr;
  for (int i=0; i<numTestCases; ++i) {
    tsOutMXConv[i] = theGammaMgr.GetMacXSec (theGammaData, tsInImat[i], tsInEkinConv[i], tsInLogEkinConv[i], 0); // conversion
    tsOutMXComp[i] = theGammaMgr.GetMacXSec (theGammaData, tsInImat[i], tsInEkinComp[i], tsInLogEkinComp[i], 1); // Compton
  }


#ifdef G4HepEm_CUDA_BUILD
  //
  // Perform the test case evaluations on the device
  double* tsOutOnDeviceMXConv = new double[numTestCases];
  double* tsOutOnDeviceMXComp = new double[numTestCases];
  TestMacXSecDataOnDevice (hepEmData, tsInImat, tsInEkinConv, tsInLogEkinConv, tsInEkinComp, tsInLogEkinComp, tsOutOnDeviceMXConv, tsOutOnDeviceMXComp, numTestCases);
  for (int i=0; i<numTestCases; ++i) {
    if ( std::abs( 1.0 - tsOutMXConv[i]/tsOutOnDeviceMXConv[i] ) > 1.0E-14 ) {
      isPassed = false;
      std::cerr << "\n*** ERROR:\nMacroscopic Cross Section data: G4HepEm Host vs Device (Conversion) mismatch: " << std::setprecision(16) << tsOutMXConv[i] << " != " << tsOutOnDeviceMXConv[i] << " ( i = " << i << " imat  = " << tsInImat[i] << " ekin =  " << tsInEkinConv[i] << ") " << std::endl;
      break;
    }
    if ( std::abs( 1.0 - tsOutMXComp[i]/tsOutOnDeviceMXComp[i] ) > 1.0E-14 ) {
      isPassed = false;
      std::cerr << "\n*** ERROR:\nMacroscopic Cross Section data: G4HepEm Host vs Device (Compton) mismatch: " <<  std::setprecision(16) << tsOutMXComp[i] << " != " << tsOutOnDeviceMXComp[i] << " ( i = " << i << " imat  = " << tsInImat[i] << " ekin =  " << tsInEkinConv[i] << ") " << std::endl;
      break;
    }
  }
  //
  delete [] tsOutOnDeviceMXConv;
  delete [] tsOutOnDeviceMXComp;
#endif // G4HepEm_CUDA_BUILD

  //
  // delete allocatd memeory
  delete [] tsInImat;
  delete [] tsInEkinConv;
  delete [] tsInLogEkinConv;
  delete [] tsInEkinComp;
  delete [] tsInLogEkinComp;
  delete [] tsOutMXConv;
  delete [] tsOutMXComp;

  return isPassed;
}
