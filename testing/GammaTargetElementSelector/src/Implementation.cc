
#include "Declaration.hh"

// G4HepEm includes
#include "G4HepEmData.hh"
#include "G4HepEmMaterialData.hh"
#include "G4HepEmGammaData.hh"

// the brem target element selector is implemented here
#include "G4HepEmGammaInteractionConversion.hh"

// don't worry it's just for testing
#define private public
#include "G4HepEmGammaManager.hh"

#include <vector>
#include <cmath>
#include <random>
#include <iostream>


bool TestGammaElemSelectorData ( const struct G4HepEmData* hepEmData ) {
  bool isPassed     = true;
  // number of material and kinetic energy pairs to generate and test (for each model)
  int  numTestCases = 32768;
  // set up an rng to get material indices on [0,numMatData)
  std::random_device rd;
  std::mt19937 gen(rd());
  gen.seed(0); // fix seed
  std::uniform_real_distribution<> dis(0, 1.0);
  // get ptr to the G4HepEmGammaData and G4HepEmMaterialData structures
  const G4HepEmGammaData*       theGammaData = hepEmData->fTheGammaData;
  const G4HepEmMaterialData* theMaterialData = hepEmData->fTheMaterialData;
  const int numMatData    = theMaterialData->fNumMaterialData;
  // allocate memory (host) to store the generated test cases:
  //  - the numTestCases, material index and kinetic energy combinations
  // and the results:
  //  - the numTestCases, index of target elements selected for the interaction
  //    corresponding to the test cases
  int*    tsInImat         = new int[numTestCases];
  double* tsInEkin         = new double[numTestCases];
  double* tsInLogEkin      = new double[numTestCases];
  double* tsInRngVals      = new double[numTestCases];
  int*    tsOutResElemIndx = new int[numTestCases];
  for (int i=0; i<numTestCases; ++i) {
    int imat          = (int)(dis(gen)*numMatData);
    tsInImat[i]       = imat;
    double minEKin    = theGammaData->fConvEnergyGrid[0];
    double maxEKin    = theGammaData->fConvEnergyGrid[theGammaData->fConvEnergyGridSize-1];
    double lMinEkin   = std::log(minEKin);
    double lEkinDelta = std::log(maxEKin/minEKin);
    tsInLogEkin[i]    = dis(gen)*lEkinDelta+lMinEkin;
    tsInEkin[i]       = std::exp(tsInLogEkin[i]);
    tsInRngVals[i]    = dis(gen);
    // get number of elements this material (from the currecnt material-cuts)
    // is composed of
    const struct G4HepEmMatData& theMatData = theMaterialData->fMaterialData[imat];
    const int numOfElement = theMatData.fNumOfElement;
    // NOTE: target element selector data are prepared only for materials that
    //       are composed from more than a single element!
    int targetElemIndx = 0;
    if (numOfElement > 1) {
      targetElemIndx = SelectTargetAtom( theGammaData, tsInImat[i], tsInEkin[i], tsInLogEkin[i], tsInRngVals[i]);
    }
    tsOutResElemIndx[i] = targetElemIndx;
    // check the selected element index aganst the number of elements the material is composed of
    if ( tsOutResElemIndx[i] >= numOfElement ) {
      isPassed = false;
      std::cerr << "\n*** ERROR:\nTarget Element Selector data for Gamma conversion: G4HepEm Host - target element index =  " << tsOutResElemIndx[i] << "  >=  #elements = " << numOfElement << " imat  = " << tsInImat[i] << " ekin =  " << tsInEkin[i] << " . " << std::endl;
      return isPassed;
    }
  } // end for-numTestCases

#ifdef G4HepEm_CUDA_BUILD
    //
    // Perform the test case evaluations on the device
    int* tsOutResOnDevice = new int[numTestCases];
    TestGammaElemSelectorDataOnDevice (hepEmData, tsInImat, tsInEkin, tsInLogEkin, tsInRngVals, tsOutResOnDevice, numTestCases);
    for (int i=0; i<numTestCases; ++i) {
      if ( tsOutResElemIndx[i] != tsOutResOnDevice[i] ) {
        isPassed = false;
        std::cerr << "\n*** ERROR:\nTarget Element Selector data for Gamma: G4HepEm Host v.s DEVICE G4HepEm Host vs Device (Conversion) mismatch: " << tsOutResElemIndx[i] << " != " << tsOutResOnDevice[i] << " imat  = " << tsInImat[i] << " ekin =  " << tsInEkin[i] << " . " << std::endl;
        break;
      }
    }
    //
    delete [] tsOutResOnDevice;
#endif // G4HepEm_CUDA_BUILD


  //
  // delete allocatd memeory
  delete [] tsInImat;
  delete [] tsInEkin;
  delete [] tsInLogEkin;
  delete [] tsInRngVals;
  delete [] tsOutResElemIndx;

  return isPassed;
}
