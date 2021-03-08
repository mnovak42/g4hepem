#include "G4HepEmElectronData.hh"

#include <iostream>

// NOTE: allocates only the main data structure but not the dynamic members
void AllocateElectronData (struct G4HepEmElectronData** theElectronData) {
  // clean away previous (if any)
  FreeElectronData(theElectronData);
  *theElectronData   = new G4HepEmElectronData;
  // eloss
  (*theElectronData)->fELossEnergyGrid                       = nullptr;
  (*theElectronData)->fELossData                             = nullptr;
  // mac-xsec
  (*theElectronData)->fResMacXSecStartIndexPerMatCut         = nullptr;
  (*theElectronData)->fResMacXSecData                        = nullptr;
  // elemen selectors per models:
  // - Moller-Bhabha ionisation
  (*theElectronData)->fElemSelectorIoniStartIndexPerMatCut   = nullptr;
  (*theElectronData)->fElemSelectorIoniData                  = nullptr;
  // - Seltzer-Berger model for e-/e+ bremsstrahlung
  (*theElectronData)->fElemSelectorBremSBStartIndexPerMatCut = nullptr;
  (*theElectronData)->fElemSelectorBremSBData                = nullptr;
  // - relativistic (improved Bethe-Heitler) model for e-/e+ bremsstrahlung
  (*theElectronData)->fElemSelectorBremRBStartIndexPerMatCut = nullptr;
  (*theElectronData)->fElemSelectorBremRBData                = nullptr;
}


void FreeElectronData (struct G4HepEmElectronData** theElectronData)  {
  if (*theElectronData) {
    // eloss
    if ((*theElectronData)->fELossEnergyGrid) {
      delete[] (*theElectronData)->fELossEnergyGrid;
    }
    if ((*theElectronData)->fELossData) {
      delete[] (*theElectronData)->fELossData;
    }
    // mac-xsec
    if ((*theElectronData)->fResMacXSecData) {
      delete[] (*theElectronData)->fResMacXSecData;
    }
    if ((*theElectronData)->fResMacXSecStartIndexPerMatCut) {
      delete[] (*theElectronData)->fResMacXSecStartIndexPerMatCut;
    }
    // element selectors:
    if ((*theElectronData)->fElemSelectorIoniStartIndexPerMatCut) {
      delete[] (*theElectronData)->fElemSelectorIoniStartIndexPerMatCut;
    }
    if ((*theElectronData)->fElemSelectorIoniData) {
      delete[] (*theElectronData)->fElemSelectorIoniData;
    }
    if ((*theElectronData)->fElemSelectorBremSBStartIndexPerMatCut) {
      delete[] (*theElectronData)->fElemSelectorBremSBStartIndexPerMatCut;
    }
    if ((*theElectronData)->fElemSelectorBremSBData) {
      delete[] (*theElectronData)->fElemSelectorBremSBData;
    }
    if ((*theElectronData)->fElemSelectorBremRBStartIndexPerMatCut) {
      delete[] (*theElectronData)->fElemSelectorBremRBStartIndexPerMatCut;
    }
    if ((*theElectronData)->fElemSelectorBremRBData) {
      delete[] (*theElectronData)->fElemSelectorBremRBData;
    }

    delete *theElectronData;
    *theElectronData = nullptr;
  }
}

#ifdef G4HepEm_CUDA_BUILD
#include <cuda_runtime.h>
#include "G4HepEmCuUtils.hh"

void CopyElectronDataToDevice(struct G4HepEmElectronData* onHOST, struct G4HepEmElectronData** onDEVICE) {
  if ( !onHOST ) return;
  // clean away previous (if any)
  if ( *onDEVICE ) {
    FreeElectronDataOnDevice ( onDEVICE );
  }
  // Create a G4HepEmElectronData structure on the host to store pointers to _d
  // side arrays on the _h side.
  struct G4HepEmElectronData* elDataHTo_d = new G4HepEmElectronData;
  // get and set number of material-cuts
  int numHepEmMatCuts = onHOST->fNumMatCuts;
  elDataHTo_d->fNumMatCuts = numHepEmMatCuts;
  //
  // === ELoss data:
  //
  const int numELossGridData = onHOST->fELossEnergyGridSize;
  // set non-pointer members  of the host side strcuture
  elDataHTo_d->fELossEnergyGridSize = numELossGridData;
  elDataHTo_d->fELossLogMinEkin     = onHOST->fELossLogMinEkin;
  elDataHTo_d->fELossEILDelta       = onHOST->fELossEILDelta;
  // allocate memory on _d for the ELoss energy grid and all ELoss data and copy
  // them from form _h
  const int numELossData = 5*numELossGridData*numHepEmMatCuts;
  gpuErrchk ( cudaMalloc ( &(elDataHTo_d->fELossEnergyGrid), sizeof( double ) * numELossGridData ) );
  gpuErrchk ( cudaMalloc ( &(elDataHTo_d->fELossData),       sizeof( double ) * numELossData     ) );
  gpuErrchk ( cudaMemcpy (   elDataHTo_d->fELossEnergyGrid,  onHOST->fELossEnergyGrid, sizeof( double ) * numELossGridData, cudaMemcpyHostToDevice ) );
  gpuErrchk ( cudaMemcpy (   elDataHTo_d->fELossData,        onHOST->fELossData,       sizeof( double ) * numELossData,     cudaMemcpyHostToDevice ) );
  //
  // === Restricted macroscopic scross section data:
  //
  // allocate memory for all the macroscopic cross section related data on _d and compy from _h
  const int numResMacXSecs = onHOST->fResMacXSecNumData;
  elDataHTo_d->fResMacXSecNumData = numResMacXSecs;
  gpuErrchk ( cudaMalloc ( &(elDataHTo_d->fResMacXSecStartIndexPerMatCut), sizeof( int )    * numHepEmMatCuts ) );
  gpuErrchk ( cudaMalloc ( &(elDataHTo_d->fResMacXSecData),                sizeof( double ) * numResMacXSecs  ) );
  gpuErrchk ( cudaMemcpy (   elDataHTo_d->fResMacXSecStartIndexPerMatCut,  onHOST->fResMacXSecStartIndexPerMatCut, sizeof( int )    * numHepEmMatCuts, cudaMemcpyHostToDevice ) );
  gpuErrchk ( cudaMemcpy (   elDataHTo_d->fResMacXSecData,                 onHOST->fResMacXSecData,                sizeof( double ) * numResMacXSecs,  cudaMemcpyHostToDevice ) );
  //
  //  === Target element selector data (for ioni and brem EM models)
  //
  // allocate memory for Ionisation related data on the _d and copy form _h
  const int numIoniData = onHOST->fElemSelectorIoniNumData;
  elDataHTo_d->fElemSelectorIoniNumData = numIoniData;
  gpuErrchk ( cudaMalloc ( &(elDataHTo_d->fElemSelectorIoniStartIndexPerMatCut), sizeof( int )    * numHepEmMatCuts ) );
  gpuErrchk ( cudaMalloc ( &(elDataHTo_d->fElemSelectorIoniData),                sizeof( double ) * numIoniData     ) );
  gpuErrchk ( cudaMemcpy (   elDataHTo_d->fElemSelectorIoniStartIndexPerMatCut,  onHOST->fElemSelectorIoniStartIndexPerMatCut, sizeof( int )    * numHepEmMatCuts, cudaMemcpyHostToDevice ) );
  gpuErrchk ( cudaMemcpy (   elDataHTo_d->fElemSelectorIoniData,                 onHOST->fElemSelectorIoniData,                sizeof( double ) * numIoniData,     cudaMemcpyHostToDevice ) );
  // the same for SB brem
  const int numBremSBData = onHOST->fElemSelectorBremSBNumData;
  elDataHTo_d->fElemSelectorBremSBNumData = numBremSBData;
  gpuErrchk ( cudaMalloc ( &(elDataHTo_d->fElemSelectorBremSBStartIndexPerMatCut), sizeof( int )    * numHepEmMatCuts ) );
  gpuErrchk ( cudaMalloc ( &(elDataHTo_d->fElemSelectorBremSBData),                sizeof( double ) * numBremSBData   ) );
  gpuErrchk ( cudaMemcpy (   elDataHTo_d->fElemSelectorBremSBStartIndexPerMatCut,  onHOST->fElemSelectorBremSBStartIndexPerMatCut, sizeof( int )    * numHepEmMatCuts, cudaMemcpyHostToDevice ) );
  gpuErrchk ( cudaMemcpy (   elDataHTo_d->fElemSelectorBremSBData,                 onHOST->fElemSelectorBremSBData,                sizeof( double ) * numBremSBData,   cudaMemcpyHostToDevice ) );
  // the same for RB brem
  const int numBremRBData = onHOST->fElemSelectorBremRBNumData;
  elDataHTo_d->fElemSelectorBremRBNumData = numBremRBData;
  gpuErrchk ( cudaMalloc ( &(elDataHTo_d->fElemSelectorBremRBStartIndexPerMatCut), sizeof( int )    * numHepEmMatCuts ) );
  gpuErrchk ( cudaMalloc ( &(elDataHTo_d->fElemSelectorBremRBData),                sizeof( double ) * numBremRBData   ) );
  gpuErrchk ( cudaMemcpy (   elDataHTo_d->fElemSelectorBremRBStartIndexPerMatCut,  onHOST->fElemSelectorBremRBStartIndexPerMatCut, sizeof( int )    * numHepEmMatCuts, cudaMemcpyHostToDevice ) );
  gpuErrchk ( cudaMemcpy (   elDataHTo_d->fElemSelectorBremRBData,                 onHOST->fElemSelectorBremRBData,                sizeof( double ) * numBremRBData,   cudaMemcpyHostToDevice ) );
  //
  // Finaly copy the top level, i.e. the main struct with the already
  // appropriate pointers to device side memory locations but stored on the host
  gpuErrchk ( cudaMalloc (  onDEVICE,              sizeof(  struct G4HepEmElectronData ) ) );
  gpuErrchk ( cudaMemcpy ( *onDEVICE, elDataHTo_d, sizeof(  struct G4HepEmElectronData ), cudaMemcpyHostToDevice ) );
  // and clean
  delete elDataHTo_d;
}


void FreeElectronDataOnDevice(struct G4HepEmElectronData** onDEVICE) {
  if (*onDEVICE) {
    // copy the on-device data back to host in order to be able to free the device
    // side dynamically allocated memories
    struct G4HepEmElectronData* onHostTo_d = new G4HepEmElectronData;
    gpuErrchk ( cudaMemcpy( onHostTo_d, *onDEVICE, sizeof( struct G4HepEmElectronData ), cudaMemcpyDeviceToHost ) );
    // ELoss data
    cudaFree( onHostTo_d->fELossEnergyGrid );
    cudaFree( onHostTo_d->fELossData       );
    // Macr. cross sections for ioni/brem
    cudaFree( onHostTo_d->fResMacXSecStartIndexPerMatCut );
    cudaFree( onHostTo_d->fResMacXSecData                );
    // Target element selectors for ioni and brem models
    cudaFree( onHostTo_d->fElemSelectorIoniStartIndexPerMatCut   );
    cudaFree( onHostTo_d->fElemSelectorIoniData                  );
    cudaFree( onHostTo_d->fElemSelectorBremSBStartIndexPerMatCut );
    cudaFree( onHostTo_d->fElemSelectorBremSBData                );
    cudaFree( onHostTo_d->fElemSelectorBremRBStartIndexPerMatCut );
    cudaFree( onHostTo_d->fElemSelectorBremRBData                );
    //
    // free the remaining device side electron data and set the host side ptr to null
    cudaFree( *onDEVICE );
    *onDEVICE = nullptr;

    delete onHostTo_d;
  }
}
#endif // G4HepEm_CUDA_BUILD
