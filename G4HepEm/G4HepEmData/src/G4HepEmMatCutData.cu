

#include "G4HepEmMatCutData.hh"
//#include <iostream>

#include <cuda_runtime.h>
#include "G4HepEmCuUtils.hh"


void CopyMatCutDataToGPU(struct G4HepEmMatCutData* onCPU, struct G4HepEmMatCutData** onGPU) {
  // clean away previous (if any)
  if ( *onGPU ) {
    FreeMatCutDataOnGPU ( onGPU );
  }
  // allocate array of G4HepEmMCCData structures on _d (its pointer adress will on _h)  
  struct G4HepEmMCCData* arrayHto_d;
  gpuErrchk ( cudaMalloc ( &arrayHto_d, sizeof( struct G4HepEmMCCData )*onCPU->fNumMatCutData ) );
  // - copy the array of G4HepEmMCCData structures from the _h to _d
  gpuErrchk ( cudaMemcpy ( arrayHto_d, onCPU->fMatCutData, onCPU->fNumMatCutData*sizeof( struct G4HepEmMCCData ), cudaMemcpyHostToDevice ) );
  // now create a helper G4HepEmMatCutData and set only its fNumMatCutData and 
  // `struct G4HepEmMCCData* fMaterialData` array member, then copy to the 
  // corresponding structure 
  struct G4HepEmMatCutData* mcData_h = new G4HepEmMatCutData;
  mcData_h->fNumMatCutData = onCPU->fNumMatCutData;
  mcData_h->fMatCutData    = arrayHto_d;
  gpuErrchk ( cudaMalloc ( onGPU, sizeof( struct G4HepEmMatCutData ) ) );
  gpuErrchk ( cudaMemcpy ( *onGPU, mcData_h, sizeof( struct G4HepEmMatCutData ), cudaMemcpyHostToDevice ) );
  // Free the auxilary G4HepEmMatCutData object
  delete mcData_h;
}

// NOTE: only the `struct G4HepEmMCCData* fMatCutData` array has been copied!
void FreeMatCutDataOnGPU ( struct G4HepEmMatCutData** onGPU ) {  
  if ( *onGPU ) {
    // copy the struct G4HepEmMatCutData` struct, including its `struct G4HepEmMCCData* fMatCutData` 
    // pointer member, from _d to _h in order to be able to free the _d sice memory 
    // pointed by `fMatCutData` by calling to cudaFree from the host.
    struct G4HepEmMatCutData* mcData_h = new G4HepEmMatCutData;
    gpuErrchk ( cudaMemcpy ( mcData_h, *onGPU, sizeof( struct G4HepEmMatCutData ), cudaMemcpyDeviceToHost ) );
    cudaFree( mcData_h->fMatCutData );
    // free the whole remaining device side memory (after cleaning all dynamically 
    // allocated members)
    cudaFree( *onGPU );
    *onGPU = nullptr;
  }
}
