#include "G4HepEmGammaData.hh"

// NOTE: allocates only the main data structure but not the dynamic members
void AllocateGammaData (struct G4HepEmGammaData** theGammaData) {
  // clean away previous (if any)
  FreeGammaData(theGammaData);
  *theGammaData   = new G4HepEmGammaData;
  // energy grids for conversion and compton
  (*theGammaData)->fConvEnergyGrid                      = nullptr;
  (*theGammaData)->fCompEnergyGrid                      = nullptr;
  // macroscopic cross sections, for conversina and compton, per materials
  (*theGammaData)->fConvCompMacXsecData                 = nullptr;  // mac-xsec
  // element selector for conversion (no need for the dummy KN compton)
  (*theGammaData)->fElemSelectorConvStartIndexPerMat    = nullptr;
  (*theGammaData)->fElemSelectorConvEgrid               = nullptr;
  (*theGammaData)->fElemSelectorConvData                = nullptr;

}


void FreeGammaData (struct G4HepEmGammaData** theGammaData)  {
  if (*theGammaData) {
    // energy grids for conversion and compton
    if ((*theGammaData)->fConvEnergyGrid ) {
      delete[] (*theGammaData)->fConvEnergyGrid ;
    }
    if ((*theGammaData)->fCompEnergyGrid) {
      delete[] (*theGammaData)->fCompEnergyGrid;
    }
    // mac-xsec for conversion and compton
    if ((*theGammaData)->fConvCompMacXsecData) {
      delete[] (*theGammaData)->fConvCompMacXsecData;
    }
    // element selector for conversion
    if ((*theGammaData)->fElemSelectorConvStartIndexPerMat) {
      delete[] (*theGammaData)->fElemSelectorConvStartIndexPerMat;
    }
    if ((*theGammaData)->fElemSelectorConvEgrid) {
      delete[] (*theGammaData)->fElemSelectorConvEgrid;
    }
    if ((*theGammaData)->fElemSelectorConvData) {
      delete[] (*theGammaData)->fElemSelectorConvData;
    }

    delete *theGammaData;
    *theGammaData = nullptr;
  }
}


#ifdef G4HepEm_CUDA_BUILD
#include <cuda_runtime.h>
#include "G4HepEmCuUtils.hh"

void CopyGammaDataToDevice(struct G4HepEmGammaData* onHOST, struct G4HepEmGammaData** onDEVICE) {/*TO BE IMPLEMENTED*/}
void FreeGammaDataOnDevice(struct G4HepEmGammaData** onDEVICE) {/*TO BE IMPLEMENTED*/}

#endif
