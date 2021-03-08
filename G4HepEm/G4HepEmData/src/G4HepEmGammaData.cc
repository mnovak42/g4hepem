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

void CopyGammaDataToDevice(struct G4HepEmGammaData* onHOST, struct G4HepEmGammaData** onDEVICE) {
  if ( !onHOST ) return;
  // clean away previous (if any)
  if ( *onDEVICE ) {
    FreeGammaDataOnDevice ( onDEVICE );
  }
  // Create a G4HepEmGammaData structure on the host to store pointers to _d
  // side arrays on the _h side.
  struct G4HepEmGammaData* gmDataHTo_d = new G4HepEmGammaData;
  // get and set number of materials
  int numHepEmMat = onHOST->fNumMaterials;
  gmDataHTo_d->fNumMaterials    = numHepEmMat;
  // -- go for the conversion related data
  int numConvData = onHOST->fConvEnergyGridSize;
  gmDataHTo_d->fConvLogMinEkin  = onHOST->fConvLogMinEkin;
  gmDataHTo_d->fConvEILDelta    = onHOST->fConvEILDelta;
  // allocate memory on _d for the conversion energy grid and copy them form _h
  gpuErrchk ( cudaMalloc ( &(gmDataHTo_d->fConvEnergyGrid), sizeof( double ) * numConvData ) );
  gpuErrchk ( cudaMemcpy (   gmDataHTo_d->fConvEnergyGrid,  onHOST->fConvEnergyGrid, sizeof( double ) * numConvData, cudaMemcpyHostToDevice ) );
  // -- go for the Compton related data
  int numCompData = onHOST->fCompEnergyGridSize;
  gmDataHTo_d->fCompLogMinEkin  = onHOST->fCompLogMinEkin;
  gmDataHTo_d->fCompEILDelta    = onHOST->fCompEILDelta;
  // allocate memory on _d for the Compton energy grid and copy them form _h
  gpuErrchk ( cudaMalloc ( &(gmDataHTo_d->fCompEnergyGrid), sizeof( double ) * numCompData ) );
  gpuErrchk ( cudaMemcpy (   gmDataHTo_d->fCompEnergyGrid,  onHOST->fCompEnergyGrid, sizeof( double ) * numCompData, cudaMemcpyHostToDevice ) );
  // allocate memory on _d for the conversion and Compton macroscopic x-section data and copy them form _h
  int numConvCompData = numHepEmMat*2*(numConvData+numCompData);
  gpuErrchk ( cudaMalloc ( &(gmDataHTo_d->fConvCompMacXsecData), sizeof( double ) * numConvCompData ) );
  gpuErrchk ( cudaMemcpy (   gmDataHTo_d->fConvCompMacXsecData,  onHOST->fConvCompMacXsecData, sizeof( double ) * numConvCompData, cudaMemcpyHostToDevice ) );
  //
  // -- go for the conversion element selector related data
  int numElSelE   = onHOST->fElemSelectorConvEgridSize;
  int numElSelDat = onHOST->fElemSelectorConvNumData;
  gmDataHTo_d->fElemSelectorConvEgridSize   = numElSelE;
  gmDataHTo_d->fElemSelectorConvNumData     = numElSelDat;
  gmDataHTo_d->fElemSelectorConvLogMinEkin  = onHOST->fElemSelectorConvLogMinEkin;
  gmDataHTo_d->fElemSelectorConvEILDelta    = onHOST->fElemSelectorConvEILDelta;
  gpuErrchk ( cudaMalloc ( &(gmDataHTo_d->fElemSelectorConvStartIndexPerMat), sizeof( int ) * numHepEmMat ) );
  gpuErrchk ( cudaMemcpy (   gmDataHTo_d->fElemSelectorConvStartIndexPerMat,  onHOST->fElemSelectorConvStartIndexPerMat, sizeof( int ) * numHepEmMat, cudaMemcpyHostToDevice ) );
  gpuErrchk ( cudaMalloc ( &(gmDataHTo_d->fElemSelectorConvEgrid), sizeof( double ) * numElSelE ) );
  gpuErrchk ( cudaMemcpy (   gmDataHTo_d->fElemSelectorConvEgrid,  onHOST->fElemSelectorConvEgrid, sizeof( double ) * numElSelE,   cudaMemcpyHostToDevice ) );
  gpuErrchk ( cudaMalloc ( &(gmDataHTo_d->fElemSelectorConvData),  sizeof( double ) * numElSelDat ) );
  gpuErrchk ( cudaMemcpy (   gmDataHTo_d->fElemSelectorConvData,   onHOST->fElemSelectorConvData,  sizeof( double ) * numElSelDat, cudaMemcpyHostToDevice ) );
  //
  // Finaly copy the top level, i.e. the main struct with the already
  // appropriate pointers to device side memory locations but stored on the host
  gpuErrchk ( cudaMalloc (  onDEVICE,              sizeof(  struct G4HepEmGammaData ) ) );
  gpuErrchk ( cudaMemcpy ( *onDEVICE, gmDataHTo_d, sizeof(  struct G4HepEmGammaData ), cudaMemcpyHostToDevice ) );
  // and clean
  delete gmDataHTo_d;
}

void FreeGammaDataOnDevice(struct G4HepEmGammaData** onDEVICE) {
  if (*onDEVICE) {
    // copy the on-device data back to host in order to be able to free the device
    // side dynamically allocated memories
    struct G4HepEmGammaData* onHostTo_d = new G4HepEmGammaData;
    gpuErrchk ( cudaMemcpy( onHostTo_d, *onDEVICE, sizeof( struct G4HepEmGammaData ), cudaMemcpyDeviceToHost ) );
    // conversion and Compton macroscopic x-section related data
    cudaFree( onHostTo_d->fConvEnergyGrid );
    cudaFree( onHostTo_d->fCompEnergyGrid );
    cudaFree( onHostTo_d->fConvCompMacXsecData );
    // conversion element selector related data
    cudaFree( onHostTo_d->fElemSelectorConvStartIndexPerMat );
    cudaFree( onHostTo_d->fElemSelectorConvEgrid );
    cudaFree( onHostTo_d->fElemSelectorConvData );
    //
    // free the remaining device side gamma data and set the host side ptr to null
    cudaFree( *onDEVICE );
    *onDEVICE = nullptr;
    // delete auxiliary object
    delete onHostTo_d;
  }
}

#endif
