

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
  G4HepEmHostDevice
  G4HepEmGammaTrack() { fTrack.ReSet(); }
  
  G4HepEmHostDevice
  G4HepEmGammaTrack(const G4HepEmGammaTrack& o) { 
    fTrack = o.fTrack;
  }

  G4HepEmHostDevice
  G4HepEmTrack*  GetTrack()  { return &fTrack; }

  G4HepEmHostDevice
  void   SetPEmxSec(double mxsec) { fPEmxSec = mxsec; }
  G4HepEmHostDevice
  double GetPEmxSec() const       { return fPEmxSec; }

  G4HepEmHostDevice
  void ReSet() {
    fTrack.ReSet();
    fPEmxSec = 0.0;
  }

    
private:
  G4HepEmTrack  fTrack;
  double        fPEmxSec;
};



#endif // G4HepEmGammaTrack_HH
