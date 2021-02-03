#include "G4HepEmSBTableData.hh"
//#include <iostream>

#include <cuda_runtime.h>
#include "G4HepEmCuUtils.hh"


void CopySBTableDataToDevice(struct G4HepEmSBTableData* onHOST, struct G4HepEmSBTableData** onDEVICE) {
  if ( !onHOST ) return;
  // clean away previous (if any)
  if ( *onDEVICE ) {
    FreeSBTableDataOnDevice ( onDEVICE );
  }
  // Create a G4HepEmSBTableData structure on the host to store pointers to _d
  // side arrays on the _h side.
  struct G4HepEmSBTableData* sbTablesHTo_d = new G4HepEmSBTableData;
  //
  // set member values
  sbTablesHTo_d->fLogMinElEnergy    = onHOST->fLogMinElEnergy;
  sbTablesHTo_d->fILDeltaElEnergy   = onHOST->fILDeltaElEnergy;
  const int numHepEmMatCuts         = onHOST->fNumHepEmMatCuts;
  const int numElemsInMatCuts       = onHOST->fNumElemsInMatCuts;
  const int numSBTableData          = onHOST->fNumSBTableData;
  sbTablesHTo_d->fNumHepEmMatCuts   = numHepEmMatCuts;
  sbTablesHTo_d->fNumElemsInMatCuts = numElemsInMatCuts;
  sbTablesHTo_d->fNumSBTableData    = numSBTableData;
  //
  // allocate device side memory for the dynamic arrys
  gpuErrchk ( cudaMalloc ( &(sbTablesHTo_d->fGammaCutIndxStartIndexPerMC), sizeof( int )    * numHepEmMatCuts   ) );
  gpuErrchk ( cudaMalloc ( &(sbTablesHTo_d->fGammaCutIndices),             sizeof( int )    * numElemsInMatCuts ) );
  gpuErrchk ( cudaMalloc ( &(sbTablesHTo_d->fSBTableData),                 sizeof( double ) * numSBTableData    ) );
  //
  gpuErrchk ( cudaMemcpy (   sbTablesHTo_d->fGammaCutIndxStartIndexPerMC,  onHOST->fGammaCutIndxStartIndexPerMC, sizeof( int )    * numHepEmMatCuts,   cudaMemcpyHostToDevice ) );
  gpuErrchk ( cudaMemcpy (   sbTablesHTo_d->fGammaCutIndices,              onHOST->fGammaCutIndices,             sizeof( int )    * numElemsInMatCuts, cudaMemcpyHostToDevice ) );
  gpuErrchk ( cudaMemcpy (   sbTablesHTo_d->fSBTableData,                  onHOST->fSBTableData,                 sizeof( double ) * numSBTableData ,   cudaMemcpyHostToDevice ) );
  //
  // Finaly copy the top level, i.e. the main struct with the already
  // appropriate pointers to device side memory locations but stored on the host
  gpuErrchk ( cudaMalloc (  onDEVICE,                sizeof(  struct G4HepEmSBTableData ) ) );
  gpuErrchk ( cudaMemcpy ( *onDEVICE, sbTablesHTo_d, sizeof(  struct G4HepEmSBTableData ), cudaMemcpyHostToDevice ) );
  // and clean
  delete sbTablesHTo_d;
}


void FreeSBTableDataOnDevice(struct G4HepEmSBTableData** onDEVICE) {
  if (*onDEVICE) {
    // copy the on-device data back to host in order to be able to free the device
    // side dynamically allocated memories
    struct G4HepEmSBTableData* onHostTo_d = new G4HepEmSBTableData;
    gpuErrchk ( cudaMemcpy( onHostTo_d, *onDEVICE, sizeof( struct G4HepEmSBTableData ), cudaMemcpyDeviceToHost ) );
    // ELoss data
    cudaFree( onHostTo_d->fGammaCutIndxStartIndexPerMC );
    cudaFree( onHostTo_d->fGammaCutIndices             );
    cudaFree( onHostTo_d->fSBTableData                 );
    //
    // free the remaining device side electron data and set the host side ptr to null
    cudaFree( *onDEVICE );
    *onDEVICE = nullptr;

    delete onHostTo_d;
  }
}
