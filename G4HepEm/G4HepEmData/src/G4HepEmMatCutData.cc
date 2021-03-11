
#include "G4HepEmMatCutData.hh"
#include <iostream>

//#include <cstdlib>

// Allocates (the only one) G4HepEmMatCutData structure
void AllocateMatCutData(struct G4HepEmMatCutData** theMatCutData, int numG4MatCuts, int numUsedG4MatCuts) {
  // clean away the previous (if any)
  FreeMatCutData ( theMatCutData );
  *theMatCutData = MakeMatCutData(numG4MatCuts, numUsedG4MatCuts);
}

G4HepEmMatCutData* MakeMatCutData(int numG4MatCuts, int numUsedG4MatCuts) {
  auto* tmp = new G4HepEmMatCutData;

  tmp->fNumG4MatCuts = numG4MatCuts;
  tmp->fG4MCIndexToHepEmMCIndex = new int[numG4MatCuts];
  // Mark all G4MC as "unused in current geometry" by default
  for ( int i=0; i<numG4MatCuts; ++i ) {
    tmp->fG4MCIndexToHepEmMCIndex[i] = -1;
  }

  tmp->fNumMatCutData = numUsedG4MatCuts;
  tmp->fMatCutData = new G4HepEmMCCData[numUsedG4MatCuts];
  return tmp;
}

// Clears (the only one) G4HepEmMatCutData structure and resets its ptr to null
void FreeMatCutData (struct G4HepEmMatCutData** theMatCutData) {
  if ( *theMatCutData != nullptr) {
    delete[] (*theMatCutData)->fG4MCIndexToHepEmMCIndex;
    delete[] (*theMatCutData)->fMatCutData;
    delete *theMatCutData;
    *theMatCutData = nullptr;
  }
}

#ifdef G4HepEm_CUDA_BUILD
#include <cuda_runtime.h>
#include "G4HepEmCuUtils.hh"

void CopyMatCutDataToGPU(struct G4HepEmMatCutData* onCPU, struct G4HepEmMatCutData** onGPU) {
  // clean away previous (if any)
  if ( *onGPU != nullptr) {
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
  if ( *onGPU != nullptr ) {
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
#endif // G4HepEm_CUDA_BUILD