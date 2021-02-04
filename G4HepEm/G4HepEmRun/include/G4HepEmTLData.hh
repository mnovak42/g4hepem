
#ifndef G4HepEmTLData_HH
#define G4HepEmTLData_HH


#include "G4HepEmElectronTrack.hh"
#include "G4HepEmGammaTrack.hh"
#include "G4HepEmRandomEngine.hh"

#include <vector>

/**
 * @file    G4HepEmTLData.hh
 * @class   G4HepEmTLData
 * @author  M. Novak
 * @date    2020
 *
 * A simple data structure to store and propagate worker local data between the components.
 *
 * Each worker `G4HepEmRunManager`-s has their own `G4HepEmTLData` object 
 * (constructed in their `G4HepEmRunManager::Initialize()` method) that is used to store: 
 *  
 *   - the thread local **random engine** object pointer: to provide independent, 
 *     unique source of random numbers for each worker when invoking the ``G4HepEm`` functions
 *   - **primary** \f$e^-/e^+\f$ and \f$\gamma\f$ **track** objects: to propagate primary track state 
 *     information to/from the (state-less) `G4HepEmElectronManager`/`G4HepEmGammaManager` functions 
 *   - **secondary** \f$e^-/e^+\f$ and \f$\gamma\f$ **track** buffers: to propagate secondary track 
 *     information (back) from the (state-less) `G4HepEmElectronManager`/`G4HepEmGammaManager` functions 
 *     as well as between these particle managers and the interaction functions
 *
 * @note
 * **All state variables** are stored in this `G4HepEmTLData` object in ``G4HepEm``.
 *
 */
 
class G4HepEmTLData {

public:
  
  G4HepEmTLData();
  
 ~G4HepEmTLData();
  
  void SetRandomEngine(G4HepEmRandomEngine* rnge) { fRNGEngine = rnge; }
  G4HepEmRandomEngine* GetRNGEngine() { return fRNGEngine; }
  
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
  G4HepEmRandomEngine*               fRNGEngine;
  
  std::size_t                        fNumSecondaryElectronTracks;
  G4HepEmElectronTrack               fElectronTrack;
  std::vector<G4HepEmElectronTrack>  fElectronSecondaryTracks;

  std::size_t                        fNumSecondaryGammaTracks;
  G4HepEmGammaTrack                  fGammaTrack;
  std::vector<G4HepEmGammaTrack>     fGammaSecondaryTracks;
  
};

#endif // G4HepEmTLData_HH
