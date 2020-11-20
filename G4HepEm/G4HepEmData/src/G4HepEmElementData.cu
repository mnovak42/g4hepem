#include "G4HepEmElementData.hh"
//#include <iostream>

#include <cuda_runtime.h>
#include "G4HepEmCuUtils.hh"


void CopyElementDataToGPU(struct G4HepEmElementData* onCPU, struct G4HepEmElementData** onGPU) {
  // clean away previous (if any)
  if ( *onGPU ) {
    FreeElementDataOnGPU ( onGPU );
  }
  // allocate array of G4HepEmElemData structures on _d (its pointer adress will on _h)  
  struct G4HepEmElemData* arrayHto_d;
  gpuErrchk ( cudaMalloc ( &arrayHto_d, sizeof( struct G4HepEmElemData )*onCPU->fMaxZet ) );
  // - copy the array of G4HepEmElemData structures from the _h to _d
  gpuErrchk ( cudaMemcpy ( arrayHto_d, onCPU->fElementData, onCPU->fMaxZet*sizeof( struct G4HepEmElemData ), cudaMemcpyHostToDevice ) );
  // now create a helper G4HepEmElementData and set its fMaxZet and 
  // `struct G4HepEmElemData* fElementData` array member, then copy to the 
  // corresponding structure 
  struct G4HepEmElementData* elData_h = new G4HepEmElementData;
  elData_h->fMaxZet      = onCPU->fMaxZet;
  elData_h->fElementData = arrayHto_d;
  gpuErrchk ( cudaMalloc ( onGPU, sizeof( struct G4HepEmElementData ) ) );
  gpuErrchk ( cudaMemcpy ( *onGPU, elData_h, sizeof( struct G4HepEmElementData ), cudaMemcpyHostToDevice ) );
  // Free the auxilary G4HepEmElementData object
  delete elData_h;
}

void FreeElementDataOnGPU ( struct G4HepEmElementData** onGPU ) {  
  if ( *onGPU ) {
    // copy the struct G4HepEmElementData` struct, including its `struct G4HepEmElemData* fElementData` 
    // pointer member, from _d to _h in order to be able to free the _d sice memory 
    // pointed by `fElementData` by calling to cudaFree from the host.
    struct G4HepEmElementData* elData_h = new G4HepEmElementData;
    gpuErrchk ( cudaMemcpy ( elData_h, *onGPU, sizeof( struct G4HepEmElementData ), cudaMemcpyDeviceToHost ) );
    cudaFree( elData_h->fElementData );
    // free the whole remaining device side memory (after cleaning all dynamically 
    // allocated members)
    cudaFree( *onGPU );
    *onGPU = nullptr;
  }
}
