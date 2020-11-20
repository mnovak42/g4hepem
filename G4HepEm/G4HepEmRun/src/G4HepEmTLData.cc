
#include "G4HepEmTLData.hh"


// g4: CLHEP include
//#include "CLHEP/Random/RandomEngine.h"

G4HepEmTLData::G4HepEmTLData() {
  fRNGEngine = nullptr;
  fElectronSecondaryTracks.resize(2);
  fNumSecondaryElectronTracks = 0;

  fGammaSecondaryTracks.resize(2);
  fNumSecondaryGammaTracks = 0;

}


G4HepEmTLData::~G4HepEmTLData() {
  // fRNGEngine is not own by this structure
}

