
#include "G4HepEmGSTableData.hh"

void AllocateGSTableData(struct G4HepEmGSTableData** theGSTableData, int numDtrData1, int numDtrData2, int numHepEmMat, int numPWACorData) {
  FreeGSTableData(theGSTableData);
  *theGSTableData = MakeGSTableData(numDtrData1, numDtrData2, numHepEmMat, numPWACorData);
}


G4HepEmGSTableData* MakeGSTableData(int numDtrData1, int numDtrData2, int numHepEmMat, int numPWACorData) {
  auto* tmp = new G4HepEmGSTableData;

  tmp->fNumDtrData1        = numDtrData1;
  tmp->fGSDtrData1         = new double[numDtrData1];
  tmp->fNumDtrData2        = numDtrData2;
  tmp->fGSDtrData2         = new double[numDtrData2];
  tmp->fNumMaterials       = numHepEmMat;
  tmp->fMoliereParams      = new double[2*numHepEmMat];
  tmp->fPWACorDataNum      = numPWACorData;
  tmp->fPWACorDataElectron = new double[numPWACorData];
  tmp->fPWACorDataPositron = new double[numPWACorData];

  return tmp;
}


void FreeGSTableData(struct G4HepEmGSTableData** theGSTableData) {
  if (*theGSTableData) {
    delete[] (*theGSTableData)->fGSDtrData1;
    delete[] (*theGSTableData)->fGSDtrData2;
    delete[] (*theGSTableData)->fMoliereParams;
    delete[] (*theGSTableData)->fPWACorDataElectron;
    delete[] (*theGSTableData)->fPWACorDataPositron;
    delete (*theGSTableData);
    *theGSTableData = nullptr;
  }
}


#ifdef G4HepEm_CUDA_BUILD
#include <cuda_runtime.h>
#include "G4HepEmCuUtils.hh"

#include <cstring>

void CopyGSTableDataToDevice(struct G4HepEmGSTableData* onHOST, struct G4HepEmGSTableData** onDEVICE) {
  if ( !onHOST ) return;
  // clean away previous (if any)
  if ( *onDEVICE ) {
    FreeGSTableDataOnDevice ( onDEVICE );
  }
  // Create a G4HepEmGSTableData structure on the host to store pointers to _d
  // side arrays on the _h side.
  struct G4HepEmGSTableData* gsTablesHTo_d = new G4HepEmGSTableData;
  // Set non-pointer members via a memcpy of the entire structure.
  memcpy(gsTablesHTo_d, onHOST, sizeof(G4HepEmGSTableData));
  const int numMaterials         = onHOST->fNumMaterials;
  const int numDtrData1          = onHOST->fNumDtrData1;
  const int numDtrData2          = onHOST->fNumDtrData2;
  const int numPWACorData        = onHOST->fPWACorDataNum;
  //
  // allocate device side memory for the dynamic arrys
  gpuErrchk ( cudaMalloc ( &(gsTablesHTo_d->fGSDtrData1),    sizeof( double ) * numDtrData1    ) );
  gpuErrchk ( cudaMalloc ( &(gsTablesHTo_d->fGSDtrData2),    sizeof( double ) * numDtrData2    ) );
  gpuErrchk ( cudaMalloc ( &(gsTablesHTo_d->fMoliereParams), sizeof( double ) * 2*numMaterials ) );
  gpuErrchk ( cudaMalloc ( &(gsTablesHTo_d->fPWACorDataElectron), sizeof( double ) * numPWACorData ) );
  gpuErrchk ( cudaMalloc ( &(gsTablesHTo_d->fPWACorDataPositron), sizeof( double ) * numPWACorData ) );

  //
  gpuErrchk ( cudaMemcpy (   gsTablesHTo_d->fGSDtrData1,    onHOST->fGSDtrData1,    sizeof( double ) * numDtrData1,    cudaMemcpyHostToDevice ) );
  gpuErrchk ( cudaMemcpy (   gsTablesHTo_d->fGSDtrData2,    onHOST->fGSDtrData2,    sizeof( double ) * numDtrData2,    cudaMemcpyHostToDevice ) );
  gpuErrchk ( cudaMemcpy (   gsTablesHTo_d->fMoliereParams, onHOST->fMoliereParams, sizeof( double ) * 2*numMaterials, cudaMemcpyHostToDevice ) );
  gpuErrchk ( cudaMemcpy (   gsTablesHTo_d->fPWACorDataElectron, onHOST->fPWACorDataElectron, sizeof( double ) * numPWACorData, cudaMemcpyHostToDevice ) );
  gpuErrchk ( cudaMemcpy (   gsTablesHTo_d->fPWACorDataPositron, onHOST->fPWACorDataPositron, sizeof( double ) * numPWACorData, cudaMemcpyHostToDevice ) );

  //
  // Finaly copy the top level, i.e. the main struct with the already
  // appropriate pointers to device side memory locations but stored on the host
  gpuErrchk ( cudaMalloc (  onDEVICE,                sizeof(  struct G4HepEmGSTableData ) ) );
  gpuErrchk ( cudaMemcpy ( *onDEVICE, gsTablesHTo_d, sizeof(  struct G4HepEmGSTableData ), cudaMemcpyHostToDevice ) );
  // and clean
  delete gsTablesHTo_d;
}


void FreeGSTableDataOnDevice(struct G4HepEmGSTableData** onDEVICE) {
  if (*onDEVICE) {
    // copy the on-device data back to host in order to be able to free the device
    // side dynamically allocated memories
    struct G4HepEmGSTableData* onHostTo_d = new G4HepEmGSTableData;
    gpuErrchk ( cudaMemcpy( onHostTo_d, *onDEVICE, sizeof( struct G4HepEmGSTableData ), cudaMemcpyDeviceToHost ) );
    //
    cudaFree( onHostTo_d->fGSDtrData1 );
    cudaFree( onHostTo_d->fGSDtrData2 );
    cudaFree( onHostTo_d->fMoliereParams );
    cudaFree( onHostTo_d->fPWACorDataElectron );
    cudaFree( onHostTo_d->fPWACorDataPositron );
    //
    // free the remaining device side electron data and set the host side ptr to null
    cudaFree( *onDEVICE );
    *onDEVICE = nullptr;

    delete onHostTo_d;
  }
}
#endif // G4HepEm_CUDA_BUILD
