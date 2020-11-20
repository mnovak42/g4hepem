

#ifndef G4HepEmGammaTrack_HH
#define G4HepEmGammaTrack_HH


#include "G4HepEmTrack.hh"

// A simple track structure for gamma particles.
//
// A G4HepEmGammaTrack is practically the NEUTRAL G4HepEmTrack object.
// Such G4HepEmGammaTrack object buffers are stored in each G4HepEmTLData 
// object, that are unique for each worker, to propagate and communicate 
// primary/secondary gamma track information between the different phases of a 
// given step or between the G4HepEmElectron/GammaManager and their functions.

class G4HepEmGammaTrack {

public:
  G4HepEmGammaTrack() { fTrack.ReSet(); }
  
  G4HepEmGammaTrack(const G4HepEmGammaTrack& o) { 
    fTrack = o.fTrack;
  }

  G4HepEmTrack*  GetTrack()  { return &fTrack; }

  void ReSet() {
    fTrack.ReSet();
  }

    
private:
  G4HepEmTrack  fTrack;
};



#endif // G4HepEmGammaTrack_HH