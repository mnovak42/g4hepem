
#ifndef G4HepEmCLHEPRandomEngine_HH
#define G4HepEmCLHEPRandomEngine_HH

#include "G4HepEmRandomEngine.hh"

#include "CLHEP/Random/RandomEngine.h"

/**
 * @file    G4HepEmCLHEPRandomEngine.hh
 * @class   G4HepEmCLHEPRandomEngine
 * @author  J. Hahnfeld
 * @date    2021
 *
 * A wrapper around CLHEP::HepRandomEngine.
 *
 * Dispatches to the virtual functions defined in the derived class.
 */
class G4HepEmCLHEPRandomEngine : public G4HepEmRandomEngine {
  // Wrapper functions to call into CLHEP::HepRandomEngine.
  static double flatWrapper(void *object) {
    return ((CLHEP::HepRandomEngine*)object)->flat();
  }
  static void flatArrayWrapper(void *object, const int size, double* vect) {
    ((CLHEP::HepRandomEngine*)object)->flatArray(size, vect);
  }

public:
  G4HepEmCLHEPRandomEngine(CLHEP::HepRandomEngine* engine)
    : G4HepEmRandomEngine(/*object=*/engine, &flatWrapper, &flatArrayWrapper) {}
};

#endif // G4HepEmCLHEPRandomEngine_HH
