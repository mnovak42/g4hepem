
#ifndef G4HepEmElectronInteractionIoni_HH
#define G4HepEmElectronInteractionIoni_HH

#include "G4HepEmConstants.hh"
#include "G4HepEmMacros.hh"
#include "G4HepEmMath.hh"
#include "G4HepEmRunUtils.hh"

class  G4HepEmTLData;
struct G4HepEmData;

// Ionisation interaction for e-/e+ described by the Moller/Bhabha model.
// Used between 100 eV - 100 TeV primary e-/e+ kinetic energies.
class G4HepEmElectronInteractionIoni {
private:
  G4HepEmElectronInteractionIoni() = delete;

public:
  static void Perform(G4HepEmTLData* tlData, struct G4HepEmData* hepEmData, bool iselectron);

  // Sampling of the energy transferred to the secondary electron in case of e-
  // primary i.e. in case of Moller interaction.
  template <typename RandomEngine>
  G4HepEmHostDevice
  static double SampleETransferMoller(const double elCut, const double primEkin, RandomEngine* rnge) {
    const double tmin    = elCut;
    const double tmax    = 0.5*primEkin;
    const double xmin    = tmin / primEkin;
    const double xmax    = tmax / primEkin;
    const double gamma   = primEkin * kInvElectronMassC2 + 1.0;
    const double gamma2  = gamma * gamma;
    const double xminmax = xmin * xmax;
    // Moller (e-e-) scattering
    const double gg      = (2.0 * gamma - 1.0) / gamma2;
    const double y       = 1. - xmax;
    const double gf      = 1.0 - gg * xmax + xmax * xmax * (1.0 - gg + (1.0 - gg * y) / (y * y));
    //
    double dum;
    double rndArray[2];
    double deltaEkin  = 0.;
    do {
      rnge->flatArray(2, rndArray);
      deltaEkin       = xminmax / (xmin * (1.0 - rndArray[0]) + xmax * rndArray[0]);
      const double xx = 1.0 - deltaEkin;
      dum             = 1.0 - gg * deltaEkin + deltaEkin * deltaEkin * (1.0 - gg + (1.0 - gg * xx) / (xx * xx));
    } while (gf * rndArray[1] > dum);
    return deltaEkin * primEkin;
  }

  // Sampling of the energy transferred to the secondary electron in case of e+
  // primary i.e. in case of Bhabha interaction.

  template <typename RandomEngine>
  G4HepEmHostDevice
  static double SampleETransferBhabha(const double elCut, const double primEkin, RandomEngine* rnge) {
    const double tmin    = elCut;
    const double tmax    = primEkin;
    const double xmin    = tmin / primEkin;
    const double xmax    = tmax / primEkin;
    const double gamma   = primEkin * kInvElectronMassC2 + 1.0;
    const double gamma2  = gamma * gamma;
    const double beta2   = 1. - 1. / gamma2;
    const double xminmax = xmin * xmax;
    // Bhabha (e+e-) scattering
    const double y       = 1.0 / (1.0 + gamma);
    const double y2      = y * y;
    const double y12     = 1.0 - 2.0 * y;
    const double b1      = 2.0 - y2;
    const double b2      = y12 * (3.0 + y2);
    const double y122    = y12 * y12;
    const double b4      = y122 * y12;
    const double b3      = b4 + y122;
    const double xmax2   = xmax * xmax;
    const double gf      = 1.0 + (xmax2 * b4 - xmin * xmin * xmin * b3 + xmax2 * b2 - xmin * b1) * beta2;
    //
    double dum;
    double rndArray[2];
    double deltaEkin  = 0.;
    do {
      rnge->flatArray(2, rndArray);
      deltaEkin       = xminmax / (xmin * (1.0 - rndArray[0]) + xmax * rndArray[0]);
      const double xx = deltaEkin * deltaEkin;
      dum             = 1.0 + (xx * xx * b4 - deltaEkin * xx * b3 + xx * b2 - deltaEkin * b1) * beta2;
    } while (gf * rndArray[1] > dum);
    return deltaEkin * primEkin;
  }

  template <typename RandomEngine>
  G4HepEmHostDevice
  static void SampleDirections(const double thePrimEkin, const double deltaEkin, double* theSecElecDir,
                               double* thePrimElecDir, RandomEngine* rnge) {
    const double elInitETot = thePrimEkin + kElectronMassC2;
    const double elInitPTot = std::sqrt(thePrimEkin * (elInitETot + kElectronMassC2));
    const double  deltaPTot = std::sqrt(deltaEkin * (deltaEkin + 2.0 * kElectronMassC2));
    const double       cost = deltaEkin * (elInitETot + kElectronMassC2) / (deltaPTot * elInitPTot);
    // check cosTheta limit
    const double   cosTheta = G4HepEmMax(-1.0, G4HepEmMin(cost, 1.0));
    const double   sinTheta = std::sqrt((1.0 - cosTheta) * (1.0 + cosTheta));
    const double        phi = k2Pi * rnge->flat();     // spherical symmetry
    //
    theSecElecDir[0]  = sinTheta * std::cos(phi);
    theSecElecDir[1]  = sinTheta * std::sin(phi);
    theSecElecDir[2]  = cosTheta;
    // rotate to refernce frame (G4HepEmRunUtils function) to get it in lab. frame
    RotateToReferenceFrame(theSecElecDir, thePrimElecDir);
    // go for the post-interaction primary electron/positiorn direction in lab. farme
    // (compute from momentum vector conservation)
    thePrimElecDir[0] = elInitPTot * thePrimElecDir[0] - deltaPTot * theSecElecDir[0];
    thePrimElecDir[1] = elInitPTot * thePrimElecDir[1] - deltaPTot * theSecElecDir[1];
    thePrimElecDir[2] = elInitPTot * thePrimElecDir[2] - deltaPTot * theSecElecDir[2];
    // normalisation
    const double  norm = 1.0 / std::sqrt(thePrimElecDir[0] * thePrimElecDir[0] + thePrimElecDir[1] * thePrimElecDir[1] + thePrimElecDir[2] * thePrimElecDir[2]);
    thePrimElecDir[0] *= norm;
    thePrimElecDir[1] *= norm;
    thePrimElecDir[2] *= norm;
  }
};

#endif // G4HepEmElectronInteractionIoni_HH
