#ifndef G4HepEmGammaInteractionCompton_HH
#define G4HepEmGammaInteractionCompton_HH

#include "G4HepEmConstants.hh"
#include "G4HepEmMacros.hh"
#include "G4HepEmMath.hh"
#include "G4HepEmRunUtils.hh"

class  G4HepEmTLData;
class  G4HepEmRandomEngine;
struct G4HepEmData;


// Compton scattering for gamma described by the simple Klein-Nishina model.
// Used between 100 eV - 100 TeV primary gamma kinetic energies.
class G4HepEmGammaInteractionCompton {
private:
  G4HepEmGammaInteractionCompton() = delete;

public:
  static void Perform(G4HepEmTLData* tlData, struct G4HepEmData* hepEmData);

  // Sampling of the post interaction photon energy and direction (already in the lab. frame)
  template <typename RandomEngine>
  G4HepEmHostDevice
  static double SamplePhotonEnergyAndDirection(const double thePrimGmE, double* thePrimGmDir,
                                               const double* theOrgPrimGmDir, RandomEngine* rnge) {
    // sample the post interaction reduced photon energy according to the KN DCS
    const double kappa = thePrimGmE * kInvElectronMassC2;
    const double eps0  = 1. / (1. + 2. * kappa);
    const double eps02 = eps0 * eps0;
    const double al1   = -G4HepEmLog(eps0);
    const double al2   = al1 + 0.5 * (1. - eps02);
    double eps, eps2, gf;
    double oneMinusCost, sint2;
    double rndm[3];
    do {
      rnge->flatArray(3, rndm);
      if (al1 > al2*rndm[0]) {
        eps  = G4HepEmExp(-al1 * rndm[1]);
        eps2 = eps * eps;
      } else {
        eps2 = eps02 + (1. - eps02) * rndm[1];
        eps  = std::sqrt(eps2);
      }
      oneMinusCost = (1. - eps) / (eps * kappa);
      sint2    = oneMinusCost * (2. - oneMinusCost);
      gf       = 1. - eps * sint2 / (1. + eps2);
    } while (gf < rndm[2]);
    // compute the post interaction photon direction and transform to lab frame
    const double cost = 1.0 - oneMinusCost;
    const double sint = std::sqrt(G4HepEmMax(0., sint2));
    const double phi  = k2Pi * rnge->flat();
    // direction of the scattered gamma in the scattering frame
    thePrimGmDir[0]   = sint * std::cos(phi);
    thePrimGmDir[1]   = sint * std::sin(phi);
    thePrimGmDir[2]   = cost;
    // rotate to refernce frame (G4HepEmRunUtils function) to get it in lab. frame
    RotateToReferenceFrame(thePrimGmDir, theOrgPrimGmDir);
    // return with the post interaction gamma energy
    return thePrimGmE*eps;
  }
};

#endif  // G4HepEmGammaInteractionCompton_HH
