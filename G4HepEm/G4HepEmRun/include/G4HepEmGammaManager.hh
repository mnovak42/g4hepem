
#ifndef G4HepEmGammaManager_HH
#define G4HepEmGammaManager_HH

struct G4HepEmData;
struct G4HepEmParameters;
//struct G4HepEmGammaData;

class  G4HepEmTLData;

/**
 * @file    G4HepEmGammaManager.hh
 * @struct  G4HepEmGammaManager
 * @author  M. Novak
 * @date    202X
 *
 * @brief The top level run-time manager for \f$\gamma\f$ transport simulations.
 *
 * It will be the same for \f$\gamma\f$ as the G4HepEmElectronManager for e-/e+.
 */

class G4HepEmGammaManager {
  
public:
  // step length
  void   HowFar(struct G4HepEmData* hepEmData, struct G4HepEmParameters* hepEmPars, G4HepEmTLData* tlData) {}
  // interactions
  void   Perform(struct G4HepEmData* hepEmData, struct G4HepEmParameters* hepEmPars, G4HepEmTLData* tlData) {}
};

#endif // G4HepEmGammaManager_HH

