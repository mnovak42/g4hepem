#ifndef G4HepEmState_HH
#define G4HepEmState_HH

struct G4HepEmData;
struct G4HepEmParameters;

/**
 * @file    G4HepEmState.hh
 * @struct  G4HepEmState
 * Non-owning view of a `G4HepEmData` and corresponding `G4HepEmParameters` instance.
 *
 * `G4HepEmRun` often requires parameters and data to be passed to its functions. The
 * `G4HepEmState` struct provides a simple aggregate of pointers to each instance for
 * convenience in handling.
 *
 * The pointers are not owned so it is up to the client to allocate and free these,
 * or copy/delete from the device as appropriate.
**/
struct G4HepEmState {
  struct G4HepEmParameters* fParameters = nullptr; //< Pointer to `G4HepEmParameters` used by `G4HepEmData` (not owned)
  struct G4HepEmData* fData = nullptr; //< Pointer to `G4HepEmData` (not owned)
};

#endif  // G4HepEmState_HH