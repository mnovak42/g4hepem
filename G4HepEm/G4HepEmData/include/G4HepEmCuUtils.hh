#ifndef G4HepEmCuUtils_HH
#define G4HepEmCuUtils_HH

#ifdef G4HepEm_CUDA_BUILD

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

/**
 * @file    G4HepEmCuUtils.hh
 * @author  M. Novak
 * @date    2020
 *
 * A simple macro for printing out  CUDA error strings.
 */


#define gpuErrchk(ans) { cuAssert((ans), __FILE__, __LINE__); }
  inline void cuAssert(cudaError_t code, const char *file, int line, bool abort=true){
      if (code != cudaSuccess) {
          fprintf(stderr,"CUAassert: %s %s %d\n",
          cudaGetErrorString(code), file, line);
          if (abort) exit(code);
      }
  }

#endif // G4HepEm_CUDA_BUILD

#endif // G4HepEmCuUtils_HH
