
#include "G4HepEmElementData.hh"

void AllocateElementData(struct G4HepEmElementData** theElementData) {
  // clean away the previous (if any)
  FreeElementData ( theElementData );
  *theElementData = MakeElementData();
}

G4HepEmElementData* MakeElementData() {
  auto* p = new G4HepEmElementData;
  const int maxZetPlusOne = 121;
  p->fMaxZet      = maxZetPlusOne-1;
  p->fElementData = new G4HepEmElemData[maxZetPlusOne];
  return p;
}

void FreeElementData(struct G4HepEmElementData** theElementData) {
  if ( *theElementData != nullptr ) {
    if ( (*theElementData)->fElementData != nullptr ) {
      for (int i = 0; i < (*theElementData)->fMaxZet; i++) {
        delete[] (*theElementData)->fElementData[i].fSandiaEnergies;
        delete[] (*theElementData)->fElementData[i].fSandiaCoefficients;
      }
      delete[] (*theElementData)->fElementData;
    }
    delete *theElementData;
    *theElementData = nullptr;
  }
}

#ifdef G4HepEm_CUDA_BUILD
#include <cuda_runtime.h>
#include "G4HepEmCuUtils.hh"

#include <cstring>

void CopyElementDataToGPU(struct G4HepEmElementData* onCPU, struct G4HepEmElementData** onGPU) {
  // clean away previous (if any)
  if ( *onGPU != nullptr) {
    FreeElementDataOnGPU ( onGPU );
  }
  // allocate array of G4HepEmElemData structures on _d (its pointer adress will on _h)
  struct G4HepEmElemData* arrayHto_d;
  gpuErrchk ( cudaMalloc ( &arrayHto_d, sizeof( struct G4HepEmElemData )*onCPU->fMaxZet ) );
  // fill in the structures on _d by copying the G4HepEmElemData one-by-one such that:
  //  - for each G4HepEmElemData struct, first allocate the double arrays
  //    on the device and set the corresponding _h reside pointers as the struct
  //    members
  //  - copy this struct from the _h to _d: the result of the copy will contain
  //    pointers to device memeory reside on the device
  struct G4HepEmElemData* dataHtoD_h = new G4HepEmElemData;
  for (int i = 0; i < onCPU->fMaxZet; i++) {
    struct G4HepEmElemData& eData_h = onCPU->fElementData[i];
    // Set non-pointer members via a memcpy of the entire structure.
    memcpy(dataHtoD_h, &eData_h, sizeof(G4HepEmElemData));

    int numSandiaIntervals = eData_h.fNumOfSandiaIntervals;
    //
    gpuErrchk ( cudaMalloc ( &(dataHtoD_h->fSandiaEnergies), sizeof( double )*numSandiaIntervals ) );
    gpuErrchk ( cudaMemcpy ( dataHtoD_h->fSandiaEnergies, eData_h.fSandiaEnergies, sizeof( double )*numSandiaIntervals, cudaMemcpyHostToDevice ) );
    //
    gpuErrchk ( cudaMalloc ( &(dataHtoD_h->fSandiaCoefficients), sizeof( double )*4*numSandiaIntervals ) );
    gpuErrchk ( cudaMemcpy ( dataHtoD_h->fSandiaCoefficients, eData_h.fSandiaCoefficients, sizeof( double )*4*numSandiaIntervals, cudaMemcpyHostToDevice ) );
    //
    // copy this G4HepEmElemData structure to _d
    gpuErrchk ( cudaMemcpy ( &(arrayHto_d[i]), dataHtoD_h, sizeof( struct G4HepEmElemData ), cudaMemcpyHostToDevice ) );
  }

  // now create a helper G4HepEmElementData and set its fMaxZet and
  // `struct G4HepEmElemData* fElementData` array member, then copy to the
  // corresponding structure
  struct G4HepEmElementData* elData_h = new G4HepEmElementData;
  // Set non-pointer members via a memcpy of the entire structure.
  memcpy(elData_h, onCPU, sizeof(G4HepEmElementData));
  elData_h->fElementData = arrayHto_d;
  gpuErrchk ( cudaMalloc ( onGPU, sizeof( struct G4HepEmElementData ) ) );
  gpuErrchk ( cudaMemcpy ( *onGPU, elData_h, sizeof( struct G4HepEmElementData ), cudaMemcpyHostToDevice ) );
  // Free the auxilary G4HepEmElementData object
  delete elData_h;
}

void FreeElementDataOnGPU ( struct G4HepEmElementData** onGPU ) {
  if ( *onGPU != nullptr ) {
    // copy the struct G4HepEmElementData` struct, including its `struct G4HepEmElemData* fElementData`
    // pointer member, from _d to _h in order to be able to free the _d sice memory
    // pointed by `fElementData` by calling to cudaFree from the host.
    struct G4HepEmElementData* elData_h = new G4HepEmElementData;
    gpuErrchk ( cudaMemcpy ( elData_h, *onGPU, sizeof( struct G4HepEmElementData ), cudaMemcpyDeviceToHost ) );
    // Then copy each of the struct G4HepEmElemData structures of the array
    // from _d to _h in order to have their double* pointer members
    // on the host, then free the pointed device memory by using these _h side
    // pointer addresses to _d side memory locations.
    struct G4HepEmElemData* eData_h = new G4HepEmElemData;
    for (int i = 0; i < elData_h->fMaxZet; i++) {
      gpuErrchk ( cudaMemcpy ( eData_h, &(elData_h->fElementData[i]), sizeof( struct G4HepEmElemData ), cudaMemcpyDeviceToHost ) );
      cudaFree ( eData_h->fSandiaEnergies );
      cudaFree ( eData_h->fSandiaCoefficients );
    }
    // Then at the and free the whole `struct G4HepEmElemData* fElementData`
    // array (after all dynamically allocated memory is freed) by using the
    // _h side address of the _d sice memory pointer.
    cudaFree( elData_h->fElementData );
    // free the whole remaining device side memory (after cleaning all dynamically
    // allocated members)
    cudaFree( *onGPU );
    *onGPU = nullptr;
    // free auxilary objects
    delete elData_h;
    delete eData_h;
  }
}
#endif // G4HepEm_CUDA_BUILD
