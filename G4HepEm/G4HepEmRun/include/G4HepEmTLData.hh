
#ifndef G4HepEmTLData_HH
#define G4HepEmTLData_HH


#include "G4HepEmElectronTrack.hh"
#include "G4HepEmGammaTrack.hh"

// g4: CLHEP include
#include "CLHEP/Random/RandomEngine.h"


//namespace CLHEP {
//  class HepRandomEngine;  
//}


//
// A simple data structure to store and propagate worker thread local infomation.
//
// Each worker thread has their own G4HepEmTLData object (constructed in their 
// unique G4HepEmRunManager::Initilize() method) that is used to store:
//  - the thread local random engine object pointer: to provide independent, 
//    unique source of random numbers for each worker thread 
//  - primary electron/gamma track objects: to propagate primary track state 
//    information between the electron/gamma manager and its functions 
//  - secondary electron/gamma track buffers: to propagate secondary track 
//    information back from the interaction models (functions) to the electron/
//    gamma managers 

class G4HepEmTLData {

public:
  
  G4HepEmTLData();
  
 ~G4HepEmTLData();
  
  void SetRandomEngine(CLHEP::HepRandomEngine* rnge) {fRNGEngine = rnge; }
  CLHEP::HepRandomEngine* GetRNGEngine() { return fRNGEngine; }
  
  G4HepEmElectronTrack* GetPrimaryElectronTrack()   { return &fElectronTrack; }
  G4HepEmElectronTrack* AddSecondaryElectronTrack() { 
    if (fNumSecondaryElectronTracks==fElectronSecondaryTracks.size()) {
      fElectronSecondaryTracks.resize(2*fElectronSecondaryTracks.size());
    }
    return &(fElectronSecondaryTracks[fNumSecondaryElectronTracks++]);
  }
  std::size_t GetNumSecondaryElectronTrack() { return fNumSecondaryElectronTracks; }
  void        ResetNumSecondaryElectronTrack() { fNumSecondaryElectronTracks = 0; }
  G4HepEmElectronTrack* GetSecondaryElectronTrack(int indx) { return &(fElectronSecondaryTracks[indx]); }
  

  G4HepEmGammaTrack* GetPrimaryGammaTrack()   { return &fGammaTrack; }
  G4HepEmGammaTrack* AddSecondaryGammaTrack() { 
    if (fNumSecondaryGammaTracks==fGammaSecondaryTracks.size()) {
      fGammaSecondaryTracks.resize(2*fGammaSecondaryTracks.size());
    }
    return &(fGammaSecondaryTracks[fNumSecondaryGammaTracks++]);
  }
  std::size_t GetNumSecondaryGammaTrack() { return fNumSecondaryGammaTracks; }
  void        ResetNumSecondaryGammaTrack() { fNumSecondaryGammaTracks = 0; }
  G4HepEmGammaTrack* GetSecondaryGammaTrack(int indx) { return &(fGammaSecondaryTracks[indx]); }
  


private:  
  
  // needs to set to point to the RNG engine of the thread
  CLHEP::HepRandomEngine*            fRNGEngine;
  
  std::size_t                        fNumSecondaryElectronTracks;
  G4HepEmElectronTrack               fElectronTrack;
  std::vector<G4HepEmElectronTrack>  fElectronSecondaryTracks;

  std::size_t                        fNumSecondaryGammaTracks;
  G4HepEmGammaTrack                  fGammaTrack;
  std::vector<G4HepEmGammaTrack>     fGammaSecondaryTracks;
  
};

#endif // G4HepEmTLData_HH
