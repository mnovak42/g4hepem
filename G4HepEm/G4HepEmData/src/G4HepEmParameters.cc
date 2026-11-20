
#include "G4HepEmParameters.hh"

void InitG4HepEmParameters (struct G4HepEmParameters* theHepEmParams) {
  FreeG4HepEmParameters(theHepEmParams);
}


void FreeG4HepEmParameters (struct G4HepEmParameters* theHepEmParams) {
  if (theHepEmParams == nullptr) {
    return;
  }
  if (theHepEmParams->fParametersPerRegion != nullptr) {
    delete[] theHepEmParams->fParametersPerRegion;
    theHepEmParams->fParametersPerRegion = nullptr;
  }

#ifdef G4HepEm_CUDA_BUILD
  FreeG4HepEmParametersOnGPU(theHepEmParams);
#endif // G4HepEm_CUDA_BUILD
}


#ifdef G4HepEm_CUDA_BUILD
#include "G4HepEmCuUtils.hh"

void CopyG4HepEmParametersToGPU(struct G4HepEmParameters* onCPU) {
  if (onCPU == nullptr) return;
  // clean away previous (if any)
  FreeG4HepEmParametersOnGPU (onCPU);
  // allocate memory on device for the G4HepEmRegionParmeters array
  gpuErrchk ( G4HepEmDeviceAPI(Malloc) ( &(onCPU->fParametersPerRegion_gpu), sizeof( struct G4HepEmRegionParmeters )*onCPU->fNumRegions ) );
  // copy the `fParametersPerRegion` G4HepEmRegionParmeters array from host to device
  gpuErrchk ( G4HepEmDeviceAPI(Memcpy) ( onCPU->fParametersPerRegion_gpu, onCPU->fParametersPerRegion, sizeof( struct G4HepEmRegionParmeters )*onCPU->fNumRegions, G4HepEmDeviceAPI(MemcpyHostToDevice) ) );
}

void FreeG4HepEmParametersOnGPU(struct G4HepEmParameters* onHost) {
  if (onHost != nullptr && onHost->fParametersPerRegion_gpu != nullptr) {
    gpuErrchk ( G4HepEmDeviceAPI(Free) (onHost->fParametersPerRegion_gpu) );
    onHost->fParametersPerRegion_gpu = nullptr;
  }
}
#endif // G4HepEm_CUDA_BUILD
