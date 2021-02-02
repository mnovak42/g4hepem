

#ifndef G4HepEmElectronTrack_HH
#define G4HepEmElectronTrack_HH


#include "G4HepEmMacros.hh"
#include "G4HepEmTrack.hh"

// A simple track structure for e-/e+ particles.
//
// A G4HepEmElectronTrack contains several extra information beyond those base 
// properties that are already available in the NEUTRAL G4HepEmTrack object.
// Such G4HepEmElectronTrack object buffers are stored in each G4HepEmTLData 
// object, that are unique for each worker, to propagate and communicate 
// primary/secondary e-/e+ track information between the different phases of a 
// given step or between the G4HepEmElectron/GammaManager and their functions.

class G4HepEmElectronTrack {

public:
  G4HepEmHostDevice
  G4HepEmElectronTrack() {     
    fTrack.ReSet(); 
    fTrack.SetCharge(-1.0);
    fRange         =  0.0;
    fPStepLength   =  0.0;
  }
  
  G4HepEmHostDevice
  G4HepEmElectronTrack(const G4HepEmElectronTrack& o) { 
    fTrack         = o.fTrack;
    fTrack.SetCharge(o.GetCharge());
    fRange         = o.fRange;
    fPStepLength   = o.fPStepLength;
  }
  
  G4HepEmHostDevice
  G4HepEmTrack*  GetTrack()  { return &fTrack; }

  G4HepEmHostDevice
  double  GetCharge() const { return fTrack.GetCharge(); }
  
  G4HepEmHostDevice
  void    SetRange(double r) { fRange = r; }
  G4HepEmHostDevice
  double  GetRange()         { return fRange; }
  
  G4HepEmHostDevice
  void    SetPStepLength(double psl)   { fPStepLength = psl;  }
  G4HepEmHostDevice
  double  GetPStepLength()             { return fPStepLength; }
  
  // Reset all member values
  G4HepEmHostDevice
  void ReSet() {
    fTrack.ReSet();
    fTrack.SetCharge(-1.0);
    fRange         = 0.0;
    fPStepLength   = 0.0;
  }
  
private:
  G4HepEmTrack  fTrack;
  double        fRange;
  double        fPStepLength;  // physical step length >= fTrack.fGStepLength
};



#endif // G4HepEmElectronTrack_HH
