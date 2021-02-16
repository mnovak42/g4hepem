
#include "G4HepEmMaterialData.hh"
#include <iostream>

// Allocates (the only one) G4HepEmMaterialData structure
void AllocateMaterialData(struct G4HepEmMaterialData** theMatData,  int numG4Mat, int numUsedG4Mat) {
  // clean away the previous (if any)
  FreeMaterialData ( theMatData );
  *theMatData = new G4HepEmMaterialData;
  (*theMatData)->fNumG4Material             = numG4Mat;
  (*theMatData)->fNumMaterialData           = numUsedG4Mat;
  (*theMatData)->fG4MatIndexToHepEmMatIndex = new int[numG4Mat];
  (*theMatData)->fMaterialData              = new G4HepEmMatData[numUsedG4Mat];
  // init G4Mat index to HepEmMat index translator to -1 (i.e. to `not used in the cur. geom.`)
  for ( int i=0; i<numG4Mat; ++i ) {
    (*theMatData)->fG4MatIndexToHepEmMatIndex[i] = -1;
  }
}


// Clears (the only one) G4HepEmMaterialData structure and resets its ptr to null
void FreeMaterialData (struct G4HepEmMaterialData** theMatData) {
  if ( *theMatData ) {
    if ( (*theMatData)->fG4MatIndexToHepEmMatIndex ) {
      delete[] (*theMatData)->fG4MatIndexToHepEmMatIndex;
    }
    if ( (*theMatData)->fMaterialData ) {
      for (int imd=0; imd<(*theMatData)->fNumMaterialData; ++imd) {
        delete[] (*theMatData)->fMaterialData[imd].fNumOfAtomsPerVolumeVect;
        delete[] (*theMatData)->fMaterialData[imd].fElementVect;
      }
      delete[] (*theMatData)->fMaterialData;
    }
    delete *theMatData;
    *theMatData = nullptr;
  }
}

#ifdef G4HepEm_CUDA_BUILD
#include <cuda_runtime.h>
#include "G4HepEmCuUtils.hh"

void CopyMaterialDataToGPU(struct G4HepEmMaterialData* onCPU, struct G4HepEmMaterialData** onGPU) {
  // clean away previous (if any)
  FreeMaterialDataOnGPU ( onGPU );
  //
  int numMatData = onCPU->fNumMaterialData;
  // allocate array of G4HepEmMatData structures on _d (its pointer address will on _h)
  struct G4HepEmMatData* arrayHto_d;
  gpuErrchk ( cudaMalloc ( &arrayHto_d, sizeof( struct G4HepEmMatData )*numMatData ) );
  // fill in the structures on _d by copying the G4HepEmMatData one-by-one such that:
  //  - for each G4HepEmMatData struct, first allocate the int and double arrays
  //    on the device and set the corresponding _h reside pointers as the struct
  //    members (possile other members can be set by value such as fNumOfElement
  //    or fDensity)
  //  - copy this struct from the _h to _d: the result of the copy will contain
  //    poinetrs to device memeory reside on the device
  struct G4HepEmMatData* dataHtoD_h = new G4HepEmMatData;
  for (int imd=0; imd<numMatData; ++imd) {
    struct G4HepEmMatData& mData_h = onCPU->fMaterialData[imd];
    int numElem = mData_h.fNumOfElement;
    dataHtoD_h->fG4MatIndex       = mData_h.fG4MatIndex;
    dataHtoD_h->fNumOfElement     = mData_h.fNumOfElement;
    dataHtoD_h->fDensity          = mData_h.fDensity;
    dataHtoD_h->fDensityCorFactor = mData_h.fDensityCorFactor;
    dataHtoD_h->fElectronDensity  = mData_h.fElectronDensity;
    dataHtoD_h->fRadiationLength  = mData_h.fRadiationLength;
    //
    gpuErrchk ( cudaMalloc ( &(dataHtoD_h->fElementVect), sizeof( int )*numElem ) );
    gpuErrchk ( cudaMemcpy ( dataHtoD_h->fElementVect, mData_h.fElementVect, sizeof( int )*numElem, cudaMemcpyHostToDevice ) );
    //
    gpuErrchk ( cudaMalloc ( &(dataHtoD_h->fNumOfAtomsPerVolumeVect), sizeof( double)*numElem ) );
    gpuErrchk ( cudaMemcpy ( dataHtoD_h->fNumOfAtomsPerVolumeVect, mData_h.fNumOfAtomsPerVolumeVect, sizeof( double )*numElem, cudaMemcpyHostToDevice ) );
    //
    // copy this G4HepEmMatData structure to _d
    gpuErrchk ( cudaMemcpy ( &(arrayHto_d[imd]), dataHtoD_h, sizeof( struct G4HepEmMatData ), cudaMemcpyHostToDevice ) );
  }
  // now create a helper G4HepEmMaterialData and set only its fNumMaterialData and
  // `struct G4HepEmMatData* fMaterialData` array member, then copy to the
  // corresponding structure from _h to _d
  struct G4HepEmMaterialData* matData_h = new G4HepEmMaterialData;
  matData_h->fNumMaterialData = numMatData;
  matData_h->fMaterialData    = arrayHto_d;
  gpuErrchk ( cudaMalloc ( onGPU, sizeof( struct G4HepEmMaterialData ) ) );
  gpuErrchk ( cudaMemcpy ( *onGPU, matData_h, sizeof( struct G4HepEmMaterialData ), cudaMemcpyHostToDevice ) );
  // celete all helper object allocated
  delete dataHtoD_h;
  delete matData_h;
}

//
void FreeMaterialDataOnGPU ( struct G4HepEmMaterialData** onGPU) {
  if ( *onGPU ) {
      // NOTE:
      // - (*onGPU) is a pointer to device memory while onGPU (i.e. struct G4HepEmMaterialData**)
      //   is the address of this pointer that is located on the host memory
      // - in order to be able to free dynamically allocated array members, such as
      //   the `G4HepEmMaterialData::fNumMaterialData` which is a type of
      //   `struct G4HepEmMatData*`, first we need to copy the address of that
      //   pointer from the device to the host. Then we can call cudaFree from
      //   the host to device pointer just copied. The same applies if we want to
      //   access any struct of struct member pointers.
      //
      // So first copy the struct G4HepEmMaterialData* from _d to _h in order
      // to have (1) _h side access to the `struct G4HepEmMatData*` array pointer
      // member and to the (2) fNumMaterialData (int) member.
      struct G4HepEmMaterialData* matData_h = new G4HepEmMaterialData;
      gpuErrchk ( cudaMemcpy ( matData_h, *onGPU, sizeof( struct G4HepEmMaterialData ), cudaMemcpyDeviceToHost ) );
      int mumMaterialData = matData_h->fNumMaterialData;
      // Then copy each of the struct G4HepEmMatData structures of the array
      // from _d to _h in order to have their int* and double* pointer members
      // on the host, then free the pointed device memory by using these _h side
      // pointer addresses to _d side memory locations.
      struct G4HepEmMatData* mData_h = new G4HepEmMatData;
      for (int imd=0; imd<mumMaterialData; ++imd) {
        gpuErrchk ( cudaMemcpy ( mData_h, &(matData_h->fMaterialData[imd]), sizeof( struct G4HepEmMatData ), cudaMemcpyDeviceToHost ) );
        cudaFree ( mData_h->fElementVect );
        cudaFree ( mData_h->fNumOfAtomsPerVolumeVect );
      }
      // Then at the and free the whole `struct G4HepEmMatData* fMaterialData`
      // array (after all dynamically allocated memory is freed) by using the
      // _h side address of the _d sice memory pointer.
      cudaFree ( matData_h->fMaterialData );
//    }
    // At the very end, we can free the whole struct.
    cudaFree( *onGPU );
    *onGPU = nullptr;
    // free auxilary objects
    delete matData_h;
    delete mData_h;
  }
}


// The oter way would be to flatten the data structures:
// 1. Should make a single array of all fElementVect and an other from all
//    fNumOfAtomsPerVolumeVect
// 2. Keep track in the material data (with a single int instead of the arrays)
//    the index where the data, that related to the given material, starts in
//    these common, flat arrays.
// This will be used in case of more important data structures such as lambda
// and loss tables.
#endif // G4HepEm_CUDA_BUILD