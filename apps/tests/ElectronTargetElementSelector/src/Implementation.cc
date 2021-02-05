
#include "Declaration.hh"

// G4HepEm includes
#include "G4HepEmData.hh"
#include "G4HepEmParameters.hh"
#include "G4HepEmMatCutData.hh"
#include "G4HepEmMaterialData.hh"
#include "G4HepEmElectronData.hh"

// the brem target element selector is implemented here
#include "G4HepEmElectronInteractionBrem.hh"

// don't worry it's just for testing
#define private public
#include "G4HepEmElectronManager.hh"

#include <vector>
#include <cmath>
#include <random>
#include <iostream>


bool TestElemSelectorData ( const struct G4HepEmData* hepEmData, const struct G4HepEmParameters* hepEmParams, bool iselectron ) {
  bool isPassed     = true;
  // number of mat-cut and kinetic energy pairs to generate and test (for each model)
  int  numTestCases = 32768;
  // set up an rng to get mc-indices on [0,numMCData)
  std::random_device rd;
  std::mt19937 gen(rd());
  gen.seed(0); // fix seed
  std::uniform_real_distribution<> dis(0, 1.0);
  // get ptr to the G4HepEmElectronData and G4HepEmMatCutData structures
  const G4HepEmElectronData* theElectronData = iselectron ? hepEmData->fTheElectronData : hepEmData->fThePositronData;
  const G4HepEmMatCutData*   theMatCutData   = hepEmData->fTheMatCutData;
  const G4HepEmMaterialData* theMaterialData = hepEmData->fTheMaterialData;
  const int numMCData    = theMatCutData->fNumMatCutData;
  // allocate memory (host) to store the generated test cases:
  //  - the numTestCases, material-cut index and kinetic energy combinations
  // and the results:
  //  - the numTestCases, index of target elements selected for the interaction
  //    corresponding to the test cases
  int*    tsInImc          = new int[numTestCases];
  double* tsInEkin         = new double[numTestCases];
  double* tsInLogEkin      = new double[numTestCases];
  double* tsInRngVals      = new double[numTestCases];
  int*    tsOutResElemIndx = new int[numTestCases];
  // model index: 0 - Moller-Bhabha ioni; 1 - Seltzer-Berger brem; 2 - Rel. brem
  // NOTE: we do not test MB model (element selector is not used)
  for (int iModel=1; iModel<3; ++iModel) {
    for (int i=0; i<numTestCases; ) {
      int imc         = (int)(dis(gen)*numMCData);
      tsInImc[i]      = imc;
      double minEKin  = 0.0;
      double maxEKin  = 0.0;
      switch (iModel) {
        case 0: { // Moller-Bhabha ioni
          double secElCutE = theMatCutData->fMatCutData[imc].fSecElProdCutE;
          minEKin = iselectron ? secElCutE : secElCutE;
          maxEKin = hepEmParams->fMaxLossTableEnergy;
          break;
        }
        case 1: {// Seltzer-Berger bremsstrahlung
          minEKin = theMatCutData->fMatCutData[imc].fSecGamProdCutE;
          maxEKin = hepEmParams->fElectronBremModelLim;
          break;
        }
        case 2: { // Relativistic bremsstrahlung
          minEKin = hepEmParams->fElectronBremModelLim;
          maxEKin = hepEmParams->fMaxLossTableEnergy;
          break;
        }
        default: { // Seltzer-Berger bremsstrahlung
          minEKin = theMatCutData->fMatCutData[imc].fSecGamProdCutE;
          maxEKin = hepEmParams->fElectronBremModelLim;
        }
      }
      if (minEKin>=maxEKin) {
        continue;
      }
      double lMinEkin   = std::log(minEKin);
      double lEkinDelta = std::log(maxEKin/minEKin);
      tsInLogEkin[i]    = dis(gen)*lEkinDelta+lMinEkin;
      tsInEkin[i]       = std::exp(tsInLogEkin[i]);
      tsInRngVals[i]    = dis(gen);
      // get number of elements this material (from the currecnt material-cuts)
      // is composed of
      const int indxMaterial = theMatCutData->fMatCutData[tsInImc[i]].fHepEmMatIndex;
      const struct G4HepEmMatData& theMatData = theMaterialData->fMaterialData[indxMaterial];
      const int numOfElement = theMatData.fNumOfElement;
      // Use the appropriate interaction model fucntion to sample the target atom
      // NOTE: target element selector data are prepared only for materials (from
      // the list of material-cuts used), that are composed from more than a single
      // element!
      int targetElemIndx = 0;
      if (numOfElement > 1) {
        switch (iModel) {
          case 1: // Seltzer-Berger bremsstrahlung (use SelectTargetAtomBrem from G4HepEmElectronInteractionBremSB)
            targetElemIndx = SelectTargetAtomBrem( theElectronData, tsInImc[i], tsInEkin[i], tsInLogEkin[i], tsInRngVals[i], true);
            break;
          case 2: // Relativistic bremsstrahlung (use SelectTargetAtomBrem from G4HepEmElectronInteractionBremSB)
            targetElemIndx = SelectTargetAtomBrem( theElectronData, tsInImc[i], tsInEkin[i], tsInLogEkin[i], tsInRngVals[i], false);
            break;
        }
      }
      tsOutResElemIndx[i] = targetElemIndx;
      // check the selected element index aganst the number of elements the material is composed of
      if ( tsOutResElemIndx[i] >= numOfElement ) {
        isPassed = false;
        std::cerr << "\n*** ERROR:\nTarget Element Selector data: G4HepEm Host - target element index =  " << tsOutResElemIndx[i] << "  >=  #elements = " << numOfElement << " at  iModel = " << iModel << " imc  = " << tsInImc[i] << " ekin =  " << tsInEkin[i] << " . " << std::endl;
        return isPassed;
      }
      // increase number of good test cases (i.e. interaction is possible with the model at the seleted material-cut + primary kinetic energy combinations)
      ++i;
    } // end for-numTestCases

#ifdef G4HepEm_CUDA_BUILD
    //
    // Perform the test case evaluations on the device
    int* tsOutResOnDevice = new int[numTestCases];
    TestElemSelectorDataOnDevice (hepEmData, tsInImc, tsInEkin, tsInLogEkin, tsInRngVals, tsOutResOnDevice, numTestCases, iModel, iselectron);
    for (int i=0; i<numTestCases; ++i) {
      if ( tsOutResElemIndx[i] != tsOutResOnDevice[i] ) {
        isPassed = false;
        std::cerr << "\n*** ERROR:\nTarget Element Selector data: G4HepEm Host v.s DEVICE G4HepEm Host vs Device (Ioni) mismatch: " << tsOutResElemIndx[i] << " != " << tsOutResOnDevice[i] << " at  iModel = " << iModel << " imc  = " << tsInImc[i] << " ekin =  " << tsInEkin[i] << " . " << std::endl;
        break;
      }
    }
    //
    delete [] tsOutResOnDevice;
#endif // G4HepEm_CUDA_BUILD

  } // end for-3 i.e. over the models

  //
  // delete allocatd memeory
  delete [] tsInImc;
  delete [] tsInEkin;
  delete [] tsInLogEkin;
  delete [] tsInRngVals;
  delete [] tsOutResElemIndx;

  return isPassed;
}
