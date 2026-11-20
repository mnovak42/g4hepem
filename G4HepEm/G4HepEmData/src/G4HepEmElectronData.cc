#include "G4HepEmElectronData.hh"

#include <iostream>

// NOTE: allocates only the main data structure but not the dynamic members
void AllocateElectronData (struct G4HepEmElectronData** theElectronData) {
  // clean away previous (if any)
  FreeElectronData(theElectronData);
  *theElectronData = MakeElectronData();
}

G4HepEmElectronData* MakeElectronData() {
  // Default construction handles everything we need, but add
  // additional initialization here if required
  return new G4HepEmElectronData;
}


void FreeElectronData (struct G4HepEmElectronData** theElectronData)  {
  if (*theElectronData != nullptr) {
    delete[] (*theElectronData)->fELossEnergyGrid;
    delete[] (*theElectronData)->fELossData;
    delete[] (*theElectronData)->fResMacXSecData;
    delete[] (*theElectronData)->fENucEnergyGrid;
    delete[] (*theElectronData)->fENucMacXsecData;
    delete[] (*theElectronData)->fTr1MacXSecData;
    delete[] (*theElectronData)->fResMacXSecStartIndexPerMatCut;
    delete[] (*theElectronData)->fElemSelectorIoniStartIndexPerMatCut;
    delete[] (*theElectronData)->fElemSelectorIoniData;
    delete[] (*theElectronData)->fElemSelectorBremSBStartIndexPerMatCut;
    delete[] (*theElectronData)->fElemSelectorBremSBData;
    delete[] (*theElectronData)->fElemSelectorBremRBStartIndexPerMatCut;
    delete[] (*theElectronData)->fElemSelectorBremRBData;

    delete *theElectronData;
    *theElectronData = nullptr;
  }
}

#ifdef G4HepEm_CUDA_BUILD
#include "G4HepEmCuUtils.hh"

#include <cstring>

void CopyElectronDataToDevice(struct G4HepEmElectronData* onHOST, struct G4HepEmElectronData** onDEVICE) {
  if ( !onHOST ) return;
  // clean away previous (if any)
  if ( *onDEVICE ) {
    FreeElectronDataOnDevice ( onDEVICE );
  }
  // Create a G4HepEmElectronData structure on the host to store pointers to _d
  // side arrays on the _h side.
  struct G4HepEmElectronData* elDataHTo_d = new G4HepEmElectronData;
  // Set non-pointer members via a memcpy of the entire structure.
  memcpy(elDataHTo_d, onHOST, sizeof(G4HepEmElectronData));
  //
  // === ELoss data:
  //
  const int numHepEmMatCuts  = onHOST->fNumMatCuts;
  const int numHepEmMats     = onHOST->fNumMaterials;
  const int numELossGridData = onHOST->fELossEnergyGridSize;
  // allocate memory on _d for the ELoss energy grid and all ELoss data and copy
  // them from form _h
  const int numELossData = 5*numELossGridData*numHepEmMatCuts;
  gpuErrchk ( G4HepEmDeviceAPI(Malloc) ( &(elDataHTo_d->fELossEnergyGrid), sizeof( double ) * numELossGridData ) );
  gpuErrchk ( G4HepEmDeviceAPI(Malloc) ( &(elDataHTo_d->fELossData),       sizeof( double ) * numELossData     ) );
  gpuErrchk ( G4HepEmDeviceAPI(Memcpy) (   elDataHTo_d->fELossEnergyGrid,  onHOST->fELossEnergyGrid, sizeof( double ) * numELossGridData, G4HepEmDeviceAPI(MemcpyHostToDevice) ) );
  gpuErrchk ( G4HepEmDeviceAPI(Memcpy) (   elDataHTo_d->fELossData,        onHOST->fELossData,       sizeof( double ) * numELossData,     G4HepEmDeviceAPI(MemcpyHostToDevice) ) );
  //
  // === Restricted macroscopic scross section data:
  //
  // allocate memory for all the macroscopic cross section related data on _d and compy from _h
  const int numResMacXSecs = onHOST->fResMacXSecNumData;
  gpuErrchk ( G4HepEmDeviceAPI(Malloc) ( &(elDataHTo_d->fResMacXSecStartIndexPerMatCut), sizeof( int )    * numHepEmMatCuts ) );
  gpuErrchk ( G4HepEmDeviceAPI(Malloc) ( &(elDataHTo_d->fResMacXSecData),                sizeof( double ) * numResMacXSecs  ) );
  gpuErrchk ( G4HepEmDeviceAPI(Memcpy) (   elDataHTo_d->fResMacXSecStartIndexPerMatCut,  onHOST->fResMacXSecStartIndexPerMatCut, sizeof( int )    * numHepEmMatCuts, G4HepEmDeviceAPI(MemcpyHostToDevice) ) );
  gpuErrchk ( G4HepEmDeviceAPI(Memcpy) (   elDataHTo_d->fResMacXSecData,                 onHOST->fResMacXSecData,                sizeof( double ) * numResMacXSecs,  G4HepEmDeviceAPI(MemcpyHostToDevice) ) );
  //
  // === Electron/postron nuclear macroscopic scross section data:
  //
  // allocate memory for all the electron/positron nuclear macroscopic cross section data on _d and compy from _h
  const int numENucGridData = onHOST->fENucEnergyGridSize;
  const int numENucData     = 2*numENucGridData*numHepEmMats;
  gpuErrchk ( G4HepEmDeviceAPI(Malloc) ( &(elDataHTo_d->fENucEnergyGrid),  sizeof( double ) * numENucGridData ) );
  gpuErrchk ( G4HepEmDeviceAPI(Malloc) ( &(elDataHTo_d->fENucMacXsecData), sizeof( double ) * numENucData     ) );
  gpuErrchk ( G4HepEmDeviceAPI(Memcpy) (   elDataHTo_d->fENucEnergyGrid,   onHOST->fENucEnergyGrid,  sizeof( double ) * numENucGridData, G4HepEmDeviceAPI(MemcpyHostToDevice) ) );
  gpuErrchk ( G4HepEmDeviceAPI(Memcpy) (   elDataHTo_d->fENucMacXsecData,  onHOST->fENucMacXsecData, sizeof( double ) * numENucData,     G4HepEmDeviceAPI(MemcpyHostToDevice) ) );
  //
  // === First macroscopic transport scross section data:
  //
  // allocate memory for all the tr1-mxsec data on _d and compy from _h
  const int numTr1MacXSecs = 2 * numELossGridData * numHepEmMats;
  gpuErrchk ( G4HepEmDeviceAPI(Malloc) ( &(elDataHTo_d->fTr1MacXSecData), sizeof( double ) * numTr1MacXSecs ) );
  gpuErrchk ( G4HepEmDeviceAPI(Memcpy) (   elDataHTo_d->fTr1MacXSecData,  onHOST->fTr1MacXSecData, sizeof( double ) * numTr1MacXSecs,  G4HepEmDeviceAPI(MemcpyHostToDevice) ) );
  //
  //  === Target element selector data (for ioni and brem EM models)
  //
  // allocate memory for Ionisation related data on the _d and copy form _h
  const int numIoniData = onHOST->fElemSelectorIoniNumData;
  if (numIoniData > 0) {
    gpuErrchk ( G4HepEmDeviceAPI(Malloc) ( &(elDataHTo_d->fElemSelectorIoniStartIndexPerMatCut), sizeof( int )    * numHepEmMatCuts ) );
    gpuErrchk ( G4HepEmDeviceAPI(Malloc) ( &(elDataHTo_d->fElemSelectorIoniData),                sizeof( double ) * numIoniData     ) );
    gpuErrchk ( G4HepEmDeviceAPI(Memcpy) (   elDataHTo_d->fElemSelectorIoniStartIndexPerMatCut,  onHOST->fElemSelectorIoniStartIndexPerMatCut, sizeof( int )    * numHepEmMatCuts, G4HepEmDeviceAPI(MemcpyHostToDevice) ) );
    gpuErrchk ( G4HepEmDeviceAPI(Memcpy) (   elDataHTo_d->fElemSelectorIoniData,                 onHOST->fElemSelectorIoniData,                sizeof( double ) * numIoniData,     G4HepEmDeviceAPI(MemcpyHostToDevice) ) );
  } else {
    elDataHTo_d->fElemSelectorIoniStartIndexPerMatCut = nullptr;
    elDataHTo_d->fElemSelectorIoniData = nullptr;
  }
  // the same for SB brem
  const int numBremSBData = onHOST->fElemSelectorBremSBNumData;
  if (numBremSBData > 0) {
    gpuErrchk ( G4HepEmDeviceAPI(Malloc) ( &(elDataHTo_d->fElemSelectorBremSBStartIndexPerMatCut), sizeof( int )    * numHepEmMatCuts ) );
    gpuErrchk ( G4HepEmDeviceAPI(Malloc) ( &(elDataHTo_d->fElemSelectorBremSBData),                sizeof( double ) * numBremSBData   ) );
    gpuErrchk ( G4HepEmDeviceAPI(Memcpy) (   elDataHTo_d->fElemSelectorBremSBStartIndexPerMatCut,  onHOST->fElemSelectorBremSBStartIndexPerMatCut, sizeof( int )    * numHepEmMatCuts, G4HepEmDeviceAPI(MemcpyHostToDevice) ) );
    gpuErrchk ( G4HepEmDeviceAPI(Memcpy) (   elDataHTo_d->fElemSelectorBremSBData,                 onHOST->fElemSelectorBremSBData,                sizeof( double ) * numBremSBData,   G4HepEmDeviceAPI(MemcpyHostToDevice) ) );
  } else {
    elDataHTo_d->fElemSelectorBremSBStartIndexPerMatCut = nullptr;
    elDataHTo_d->fElemSelectorBremSBData = nullptr;
  }
  // the same for RB brem
  const int numBremRBData = onHOST->fElemSelectorBremRBNumData;
  if (numBremRBData > 0) {
    gpuErrchk ( G4HepEmDeviceAPI(Malloc) ( &(elDataHTo_d->fElemSelectorBremRBStartIndexPerMatCut), sizeof( int )    * numHepEmMatCuts ) );
    gpuErrchk ( G4HepEmDeviceAPI(Malloc) ( &(elDataHTo_d->fElemSelectorBremRBData),                sizeof( double ) * numBremRBData   ) );
    gpuErrchk ( G4HepEmDeviceAPI(Memcpy) (   elDataHTo_d->fElemSelectorBremRBStartIndexPerMatCut,  onHOST->fElemSelectorBremRBStartIndexPerMatCut, sizeof( int )    * numHepEmMatCuts, G4HepEmDeviceAPI(MemcpyHostToDevice) ) );
    gpuErrchk ( G4HepEmDeviceAPI(Memcpy) (   elDataHTo_d->fElemSelectorBremRBData,                 onHOST->fElemSelectorBremRBData,                sizeof( double ) * numBremRBData,   G4HepEmDeviceAPI(MemcpyHostToDevice) ) );
  } else {
    elDataHTo_d->fElemSelectorBremRBStartIndexPerMatCut = nullptr;
    elDataHTo_d->fElemSelectorBremRBData = nullptr;
  }
  //
  // Finaly copy the top level, i.e. the main struct with the already
  // appropriate pointers to device side memory locations but stored on the host
  gpuErrchk ( G4HepEmDeviceAPI(Malloc) (  onDEVICE,              sizeof(  struct G4HepEmElectronData ) ) );
  gpuErrchk ( G4HepEmDeviceAPI(Memcpy) ( *onDEVICE, elDataHTo_d, sizeof(  struct G4HepEmElectronData ), G4HepEmDeviceAPI(MemcpyHostToDevice) ) );
  // and clean
  delete elDataHTo_d;
}


void FreeElectronDataOnDevice(struct G4HepEmElectronData** onDEVICE) {
  if (*onDEVICE) {
    // copy the on-device data back to host in order to be able to free the device
    // side dynamically allocated memories
    struct G4HepEmElectronData* onHostTo_d = new G4HepEmElectronData;
    gpuErrchk ( G4HepEmDeviceAPI(Memcpy)( onHostTo_d, *onDEVICE, sizeof( struct G4HepEmElectronData ), G4HepEmDeviceAPI(MemcpyDeviceToHost) ) );
    // ELoss data
    gpuErrchk ( G4HepEmDeviceAPI(Free)( onHostTo_d->fELossEnergyGrid ) );
    gpuErrchk ( G4HepEmDeviceAPI(Free)( onHostTo_d->fELossData       ) );
    // Macr. cross sections for ioni/brem
    gpuErrchk ( G4HepEmDeviceAPI(Free)( onHostTo_d->fResMacXSecStartIndexPerMatCut ) );
    gpuErrchk ( G4HepEmDeviceAPI(Free)( onHostTo_d->fResMacXSecData                ) );
    // Macr. scross sections for eletron/positron nuclear and the energy grid
    gpuErrchk ( G4HepEmDeviceAPI(Free)( onHostTo_d->fENucEnergyGrid  ) );
    gpuErrchk ( G4HepEmDeviceAPI(Free)( onHostTo_d->fENucMacXsecData ) );
    // Tr1-mxsec data
    gpuErrchk ( G4HepEmDeviceAPI(Free)( onHostTo_d->fTr1MacXSecData                ) );
    // Target element selectors for ioni and brem models
    gpuErrchk ( G4HepEmDeviceAPI(Free)( onHostTo_d->fElemSelectorIoniStartIndexPerMatCut   ) );
    gpuErrchk ( G4HepEmDeviceAPI(Free)( onHostTo_d->fElemSelectorIoniData                  ) );
    gpuErrchk ( G4HepEmDeviceAPI(Free)( onHostTo_d->fElemSelectorBremSBStartIndexPerMatCut ) );
    gpuErrchk ( G4HepEmDeviceAPI(Free)( onHostTo_d->fElemSelectorBremSBData                ) );
    gpuErrchk ( G4HepEmDeviceAPI(Free)( onHostTo_d->fElemSelectorBremRBStartIndexPerMatCut ) );
    gpuErrchk ( G4HepEmDeviceAPI(Free)( onHostTo_d->fElemSelectorBremRBData                ) );
    //
    // free the remaining device side electron data and set the host side ptr to null
    gpuErrchk ( G4HepEmDeviceAPI(Free)( *onDEVICE ) );
    *onDEVICE = nullptr;

    delete onHostTo_d;
  }
}
#endif // G4HepEm_CUDA_BUILD
