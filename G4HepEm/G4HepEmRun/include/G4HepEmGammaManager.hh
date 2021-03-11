
#ifndef G4HepEmGammaManager_HH
#define G4HepEmGammaManager_HH

#include "G4HepEmMacros.hh"

struct G4HepEmData;
struct G4HepEmParameters;
struct G4HepEmGammaData;

class  G4HepEmTLData;
class  G4HepEmGammaTrack;
class  G4HepEmTrack;

/**
 * @file    G4HepEmGammaManager.hh
 * @struct  G4HepEmGammaManager
 * @author  M. Novak
 * @date    2021
 *
 * @brief The top level run-time manager for \f$\gamma\f$ transport simulations.
 *
 * It will be the same for \f$\gamma\f$ as the G4HepEmElectronManager for e-/e+.
 */

class G4HepEmGammaManager {

public:
  // step length
  void   HowFar(struct G4HepEmData* /*hepEmData*/, struct G4HepEmParameters* /*hepEmPars*/, G4HepEmTLData* /*tlData*/);

  G4HepEmHostDevice
  void   HowFar(struct G4HepEmData* hepEmData, struct G4HepEmParameters* hepEmPars, G4HepEmTrack* theTrack);


  // interactions
  void   Perform(struct G4HepEmData* /*hepEmData*/, struct G4HepEmParameters* /*hepEmPars*/, G4HepEmTLData* /*tlData*/);

  G4HepEmHostDevice
  void   UpdateNumIALeft(G4HepEmTrack* theTrack);


  G4HepEmHostDevice
  double  GetMacXSec(const struct G4HepEmGammaData* gmData, const int imat, const double ekin, const double lekin, const int iprocess);



};

#endif // G4HepEmGammaManager_HH
