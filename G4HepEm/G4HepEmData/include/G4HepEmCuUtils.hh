#ifndef G4HepEmCuUtils_HH
#define G4HepEmCuUtils_HH

#ifdef G4HepEm_CUDA_BUILD

// G4HepEm_CUDA_BUILD enables the device data; G4HepEm_HIP_BUILD selects HIP instead of CUDA for it
#ifdef G4HepEm_HIP_BUILD
#ifdef __HIP__
// Device sources (the test drivers) also need the kernel built-ins
#include <hip/hip_runtime.h>
#else
#include <hip/hip_runtime_api.h>
#endif
#define G4HepEmDeviceAPI(TOK) hip##TOK
#else
#include <cuda_runtime.h>
#define G4HepEmDeviceAPI(TOK) cuda##TOK
#endif
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
  inline void cuAssert(G4HepEmDeviceAPI(Error_t) code, const char *file, int line, bool abort=true){
      if (code != G4HepEmDeviceAPI(Success)) {
          fprintf(stderr,"CUAassert: %s %s %d\n",
          G4HepEmDeviceAPI(GetErrorString)(code), file, line);
          if (abort) exit(code);
      }
  }

#endif // G4HepEm_CUDA_BUILD

#endif // G4HepEmCuUtils_HH
