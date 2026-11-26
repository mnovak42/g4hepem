#include "G4HepEmGammaData.hh"

// NOTE: allocates only the main data structure but not the dynamic members
void AllocateGammaData (struct G4HepEmGammaData** theGammaData) {
  // clean away previous (if any)
  FreeGammaData(theGammaData);
  *theGammaData   = MakeGammaData();
}

G4HepEmGammaData* MakeGammaData() {
  // Default construction handles everything we need, but add
  // additional initialization here if required
  return new G4HepEmGammaData;
}

void FreeGammaData (struct G4HepEmGammaData** theGammaData)  {
  if (*theGammaData != nullptr) {
    delete[] (*theGammaData)->fMacXsecData;
    delete[] (*theGammaData)->fElemSelectorConvStartIndexPerMat;
    delete[] (*theGammaData)->fElemSelectorConvEgrid;
    delete[] (*theGammaData)->fElemSelectorConvData;
    delete *theGammaData;
    *theGammaData = nullptr;
  }
}


#ifdef G4HepEm_CUDA_BUILD
#include "G4HepEmCuUtils.hh"

#include <cstring>

void CopyGammaDataToDevice(struct G4HepEmGammaData* onHOST, struct G4HepEmGammaData** onDEVICE) {
  if ( !onHOST ) return;
  // clean away previous (if any)
  if ( *onDEVICE ) {
    FreeGammaDataOnDevice ( onDEVICE );
  }
  // Create a G4HepEmGammaData structure on the host to store pointers to _d
  // side arrays on the _h side.
  struct G4HepEmGammaData* gmDataHTo_d = new G4HepEmGammaData;
  // Set non-pointer members via a memcpy of the entire structure.
  memcpy(gmDataHTo_d, onHOST, sizeof(G4HepEmGammaData));
  // get and set number of macroscopic cross section data
  int numHepEmMat = onHOST->fNumMaterials;
  int numMacXsecData =  numHepEmMat * onHOST->fDataPerMat;
  // allocate memory on _d for the macroscopic cross section data and copy them form _h
  gpuErrchk ( G4HepEmDeviceAPI(Malloc) ( &(gmDataHTo_d->fMacXsecData), sizeof( double ) * numMacXsecData ) );
  gpuErrchk ( G4HepEmDeviceAPI(Memcpy) (   gmDataHTo_d->fMacXsecData,  onHOST->fMacXsecData, sizeof( double ) * numMacXsecData, G4HepEmDeviceAPI(MemcpyHostToDevice) ) );
  // -- go for the conversion element selector related data
  int numElSelE   = onHOST->fElemSelectorConvEgridSize;
  int numElSelDat = onHOST->fElemSelectorConvNumData;
  if (numElSelDat > 0) {
    gpuErrchk ( G4HepEmDeviceAPI(Malloc) ( &(gmDataHTo_d->fElemSelectorConvStartIndexPerMat), sizeof( int ) * numHepEmMat ) );
    gpuErrchk ( G4HepEmDeviceAPI(Memcpy) (   gmDataHTo_d->fElemSelectorConvStartIndexPerMat,  onHOST->fElemSelectorConvStartIndexPerMat, sizeof( int ) * numHepEmMat, G4HepEmDeviceAPI(MemcpyHostToDevice) ) );
    gpuErrchk ( G4HepEmDeviceAPI(Malloc) ( &(gmDataHTo_d->fElemSelectorConvEgrid), sizeof( double ) * numElSelE ) );
    gpuErrchk ( G4HepEmDeviceAPI(Memcpy) (   gmDataHTo_d->fElemSelectorConvEgrid,  onHOST->fElemSelectorConvEgrid, sizeof( double ) * numElSelE,   G4HepEmDeviceAPI(MemcpyHostToDevice) ) );
    gpuErrchk ( G4HepEmDeviceAPI(Malloc) ( &(gmDataHTo_d->fElemSelectorConvData),  sizeof( double ) * numElSelDat ) );
    gpuErrchk ( G4HepEmDeviceAPI(Memcpy) (   gmDataHTo_d->fElemSelectorConvData,   onHOST->fElemSelectorConvData,  sizeof( double ) * numElSelDat, G4HepEmDeviceAPI(MemcpyHostToDevice) ) );
  } else {
    gmDataHTo_d->fElemSelectorConvStartIndexPerMat = nullptr;
    gmDataHTo_d->fElemSelectorConvEgrid = nullptr;
    gmDataHTo_d->fElemSelectorConvData = nullptr;
  }
  //
  // Finaly copy the top level, i.e. the main struct with the already
  // appropriate pointers to device side memory locations but stored on the host
  gpuErrchk ( G4HepEmDeviceAPI(Malloc) (  onDEVICE,              sizeof(  struct G4HepEmGammaData ) ) );
  gpuErrchk ( G4HepEmDeviceAPI(Memcpy) ( *onDEVICE, gmDataHTo_d, sizeof(  struct G4HepEmGammaData ), G4HepEmDeviceAPI(MemcpyHostToDevice) ) );
  // and clean
  delete gmDataHTo_d;
}

void FreeGammaDataOnDevice(struct G4HepEmGammaData** onDEVICE) {
  if (*onDEVICE) {
    // copy the on-device data back to host in order to be able to free the device
    // side dynamically allocated memories
    struct G4HepEmGammaData* onHostTo_d = new G4HepEmGammaData;
    gpuErrchk ( G4HepEmDeviceAPI(Memcpy)( onHostTo_d, *onDEVICE, sizeof( struct G4HepEmGammaData ), G4HepEmDeviceAPI(MemcpyDeviceToHost) ) );
    // total, conversion, Compton, PE macroscopic x-section related data
    gpuErrchk ( G4HepEmDeviceAPI(Free)( onHostTo_d->fMacXsecData ) );
    // conversion element selector related data
    gpuErrchk ( G4HepEmDeviceAPI(Free)( onHostTo_d->fElemSelectorConvStartIndexPerMat ) );
    gpuErrchk ( G4HepEmDeviceAPI(Free)( onHostTo_d->fElemSelectorConvEgrid ) );
    gpuErrchk ( G4HepEmDeviceAPI(Free)( onHostTo_d->fElemSelectorConvData ) );
    //
    // free the remaining device side gamma data and set the host side ptr to null
    gpuErrchk ( G4HepEmDeviceAPI(Free)( *onDEVICE ) );
    *onDEVICE = nullptr;
    // delete auxiliary object
    delete onHostTo_d;
  }
}

#endif
