
#ifndef G4HepEmPositronInteractionAnnihilation_HH
#define G4HepEmPositronInteractionAnnihilation_HH

#include "G4HepEmConstants.hh"
#include "G4HepEmMacros.hh"
#include "G4HepEmMath.hh"
#include "G4HepEmRunUtils.hh"

class G4HepEmTLData;

// e+ annihilation to two gamma interaction described by the Heitler model.
// Used between 0 eV - 100 TeV primary e+ kinetic energies i.e.
// covers both in-flight and at-rest annihilation.
class G4HepEmPositronInteractionAnnihilation {
private:
  G4HepEmPositronInteractionAnnihilation() = delete;

public:
  static void Perform(G4HepEmTLData* tlData, bool isatrest);

  // e+ is already at rest case
  static void AnnihilateAtRest(G4HepEmTLData* tlData);
  // e+ is in-flight case
  static void AnnihilateInFlight(G4HepEmTLData* tlData);

  template <typename RandomEngine>
  G4HepEmHostDevice
  static void SampleEnergyAndDirectionsInFlight(const double thePrimEkin, const double *thePrimDir,
                                                double *theGamma1Ekin, double *theGamma1Dir,
                                                double *theGamma2Ekin, double *theGamma2Dir,
                                                RandomEngine* rnge) {
    // compute kinetic limits
    const double tau     = thePrimEkin*kInvElectronMassC2;
    const double gam     = tau + 1.0;
    const double tau2    = tau + 2.0;
    const double sqgrate = std::sqrt(tau/tau2)*0.5;
    //
    const double epsmin  = 0.5 - sqgrate;
    const double epsmax  = 0.5 + sqgrate;
    const double epsqot  = epsmax/epsmin;
    // sampling of the energy rate of the gammas
    const double tau4    = tau2*tau2;
    double eps   = 0.0;
    double rfunc = 0.0;
    double rndArray[2];
    do {
      rnge->flatArray(2, rndArray);
      eps   = epsmin*G4HepEmExp(G4HepEmLog(epsqot)*rndArray[0]);
      rfunc = 1. - eps + (2.*gam*eps-1.)/(eps*tau4);
    } while( rfunc < rndArray[1]);
    // compute direction of the gammas
    const double sqg2m1 = std::sqrt(tau*tau2);
    const double   cost = G4HepEmMin(1., G4HepEmMax(-1., (eps*tau2-1.)/(eps*sqg2m1)));
    const double   sint = std::sqrt((1.+cost)*(1.-cost));
    const double    phi = k2Pi * rnge->flat();
    // kinematics of the first gamma
    const double initEt = thePrimEkin + 2.*kElectronMassC2;
    const double ekinG1 = eps*initEt;
    *theGamma1Ekin = ekinG1;
    theGamma1Dir[0] = sint*std::cos(phi);
    theGamma1Dir[1] = sint*std::sin(phi);
    theGamma1Dir[2] = cost;
    // use the G4HepEmRunUtils function
    RotateToReferenceFrame(theGamma1Dir, thePrimDir);
    // kinematics of the second gamma (direction <== conservation)
    *theGamma2Ekin = initEt-ekinG1;
    const double initPt = std::sqrt(thePrimEkin*(thePrimEkin+2*kElectronMassC2));
    const double     px = initPt*thePrimDir[0] - theGamma1Dir[0]*ekinG1;
    const double     py = initPt*thePrimDir[1] - theGamma1Dir[1]*ekinG1;
    const double     pz = initPt*thePrimDir[2] - theGamma1Dir[2]*ekinG1;
    const double   norm = 1.0 / std::sqrt(px*px + py*py + pz*pz);
    theGamma2Dir[0] = px*norm;
    theGamma2Dir[1] = py*norm;
    theGamma2Dir[2] = pz*norm;
  }

};

#endif // G4HepEmPositronInteractionAnnihilation_HH
