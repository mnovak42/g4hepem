#ifndef G4HepEmCuUtils_HH
#define G4HepEmCuUtils_HH

#ifdef G4HepEm_CUDA_BUILD

#include <cuda_runtime.h>
#include <cstdio> 

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
  inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
      if (code != cudaSuccess) {
          fprintf(stderr,"GPUassert: %s %s %d\n",
          cudaGetErrorString(code), file, line);
          if (abort) exit(code);
      }   
  }

#endif // G4HepEm_CUDA_BUILD

#endif // G4HepEmCuUtils_HH
