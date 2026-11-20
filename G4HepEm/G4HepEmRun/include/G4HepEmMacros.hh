

#ifndef G4HepEmMacros_HH
#define G4HepEmMacros_HH

#if defined(__CUDACC__) || defined(__HIP__)
#define G4HepEmHostDevice __host__ __device__
#else
#define G4HepEmHostDevice
#endif

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define G4HepEm_DEVICE_COMPILE
// If compiling for the device, make the constant available.
#define G4HepEmHostDeviceConstant __device__
#else
#define G4HepEmHostDeviceConstant
#endif

#endif // G4HepEmMacros_HH
