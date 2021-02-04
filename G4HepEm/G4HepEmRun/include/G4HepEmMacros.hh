

#ifndef G4HepEmMacros_HH
#define G4HepEmMacros_HH

#ifdef __CUDACC__
#define G4HepEmHostDevice __host__ __device__
#else
#define G4HepEmHostDevice
#endif

#ifdef __CUDA_ARCH__
// If compiling for the device, make the constant available.
#define G4HepEmHostDeviceConstant __device__
#else
#define G4HepEmHostDeviceConstant
#endif

#endif // G4HepEmMacros_HH
