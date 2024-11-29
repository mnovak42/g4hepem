
#include "Declaration.hh"


// G4HepEm includes
#include "G4HepEmData.hh"
#include "G4HepEmMaterialData.hh"
#include "G4HepEmGammaData.hh"

#include "G4HepEmGammaManager.hh"
#include "G4HepEmGammaTrack.hh"

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

  const int numMatData  = theMatData->fNumMaterialData;
  // allocate memory (host) to store the generated test cases:
  //  - the numTestCases, material index, kinetic energy combinations and random
  //    number combinations
  // and the results:
  //  - the numTestCases total mac. xsec and the ID of the sampled interactions
  int*    tsInImat    = new int[numTestCases];

  double* tsInEkin    = new double[numTestCases];
  double* tsInLogEkin = new double[numTestCases];
  double* tsInURand   = new double[numTestCases];

  double* tsOutMXTot  = new double[numTestCases];
  int*    tsOutProcID = new int[numTestCases];

  for (int i=0; i<numTestCases; ++i) {
    int imat           = (int)(dis(gen)*numMatData);
    tsInImat[i]        = imat;
    // -- total macroscopic cross section (+- 2% below/above the energy grid)
    double lMinEkin    = theGammaData->fEMin0*0.98;
    double lEkinDelta  = std::log(1.02*theGammaData->fEMax2) - lMinEkin;
    tsInLogEkin[i] = dis(gen)*lEkinDelta+lMinEkin;
    tsInEkin[i]    = std::exp(tsInLogEkin[i]);

    tsInURand[i] = dis(gen);
}
  //
  // Use G4HepEmGammaManager to evaluate the macroscopic cross sections
  // for conversion inot e-e+ pairs and Compton scattering.
  G4HepEmGammaTrack aGammaTrack;
  G4HepEmTrack* aTrack = aGammaTrack.GetTrack();
  for (int i=0; i<numTestCases; ++i) {
    tsOutMXTot[i] = G4HepEmGammaManager::GetTotalMacXSec (hepEmData, tsInImat[i], tsInEkin[i], tsInLogEkin[i], &aGammaTrack); // total mxces
    // set all track fields that the sampling below needs
    const double totMFP = (tsOutMXTot[i] > 0) ? 1.0/tsOutMXTot[i] : 1E+20;
    if (tsOutMXTot[i]>0) { // otherwise IMFP would be such that we never call sampling
      aTrack->SetMFP(totMFP, 0);
      aTrack->SetEKin(tsInEkin[i]);
      aTrack->SetMCIndex(tsInImat[i]); // this is the material index but valid as mat-cut index as well
      G4HepEmGammaManager::SampleInteraction(hepEmData, &aGammaTrack, tsInURand[i]); // sample interaction
      tsOutProcID[i] = aGammaTrack.GetTrack()->GetWinnerProcessIndex();
    }
  }


#ifdef G4HepEm_CUDA_BUILD
  //
  // Perform the test case evaluations on the device
  double* tsOutOnDeviceMXConv = new double[numTestCases];
  double* tsOutOnDeviceMXComp = new double[numTestCases];
  double* tsOutOnDeviceMXGNuc = new double[numTestCases];
  TestMacXSecDataOnDevice (hepEmData, tsInImat, tsInEkinConv, tsInLogEkinConv, tsInEkinComp, tsInLogEkinComp, tsInEkinGNuc, tsInLogEkinGNuc, tsOutOnDeviceMXConv, tsOutOnDeviceMXComp, tsOutOnDeviceMXGNuc, numTestCases);
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
    if ( std::abs( 1.0 - tsOutMXGNuc[i]/tsOutOnDeviceMXGNuc[i] ) > 1.0E-14 ) {
      isPassed = false;
      std::cerr << "\n*** ERROR:\nMacroscopic Cross Section data: G4HepEm Host vs Device (Gamma-nuclear) mismatch: " <<  std::setprecision(16) << tsOutMXGNuc[i] << " != " << tsOutOnDeviceMXGNuc[i] << " ( i = " << i << " imat  = " << tsInImat[i] << " ekin =  " << tsInEkinGNuc[i] << ") " << std::endl;
      break;
    }
  }
  //
  delete [] tsOutOnDeviceMXConv;
  delete [] tsOutOnDeviceMXComp;
  delete [] tsOutOnDeviceMXGNuc;
#endif // G4HepEm_CUDA_BUILD

  //
  // delete allocatd memeory
  delete [] tsInImat;
  delete [] tsInEkin;
  delete [] tsInLogEkin;
  delete [] tsInURand;
  delete [] tsOutMXTot;
  delete [] tsOutProcID;


  return isPassed;
}
