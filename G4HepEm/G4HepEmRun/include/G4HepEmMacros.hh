

#ifndef G4HepEmMacros_HH
#define G4HepEmMacros_HH

#ifdef __CUDACC__
#define G4HepEmHostDevice __host__ __device__
#else
#define G4HepEmHostDevice
#endif

#endif // G4HepEmMacros_HH
