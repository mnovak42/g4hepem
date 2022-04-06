#ifndef G4HepEmGammaInteractionPhotoelectric_HH
#define G4HepEmGammaInteractionPhotoelectric_HH

#include "G4HepEmConstants.hh"
#include "G4HepEmData.hh"
#include "G4HepEmElementData.hh"
#include "G4HepEmMacros.hh"
#include "G4HepEmMatCutData.hh"
#include "G4HepEmMaterialData.hh"
#include "G4HepEmRunUtils.hh"

#include <cmath>

class G4HepEmTLData;

class G4HepEmGammaInteractionPhotoelectric {
private:
  G4HepEmGammaInteractionPhotoelectric() = delete;

public:
  static void Perform(G4HepEmTLData* tlData, struct G4HepEmData* hepEmData);

  template <typename RandomEngine>
  G4HepEmHostDevice
  static double SelectElementBindingEnergy(const struct G4HepEmData* hepEmData, const int imc, const double mxsec, const double ekin, RandomEngine* rnge) {
    const int theMatIndx = hepEmData->fTheMatCutData->fMatCutData[imc].fHepEmMatIndex;
    const G4HepEmMatData& theMData = hepEmData->fTheMaterialData->fMaterialData[theMatIndx];

    // Possible optimization: if ekin minus the minimum binding energy of all elements in the material
    // is already smaller than the electron cut, we could skip selecting an element.

    int ielem = 0;
    if (theMData.fNumOfElement > 1) {
      const double x = rnge->flat() * mxsec;
      double sum = 0;
      double invE = 1 / ekin;
      for (int i = 0; i < theMData.fNumOfElement; i++) {
        const G4HepEmElemData& theEData = hepEmData->fTheElementData->fElementData[theMData.fElementVect[i]];
        int interval = 0;
        if (ekin >= theEData.fSandiaEnergies[0]) {
          // Optimization: linear search starting with intervals for higher energies.
          for (int i = theEData.fNumOfSandiaIntervals - 1; i >= 0; i--) {
            if (ekin >= theEData.fSandiaEnergies[i]) {
              interval = i;
              break;
            }
          }
        }
        const double* sandiaCof = &theEData.fSandiaCoefficients[4 * interval];
        sum += theMData.fNumOfAtomsPerVolumeVect[i] *
               invE * (sandiaCof[0] + invE * (sandiaCof[1] + invE * (sandiaCof[2] + invE * sandiaCof[3])));
        if (x <= sum) {
          ielem = i;
          break;
        }
      }
    }

    // Only check the k-shell binding energy of the sampled element. If the
    // gamma energy is below that, the energy of the secondary photo electron
    // will be so low that it won't travel far and we can skip generating it.

    return hepEmData->fTheElementData->fElementData[theMData.fElementVect[ielem]].fKShellBindingEnergy;
  }


  template <typename RandomEngine>
  G4HepEmHostDevice
  static void SamplePhotoElectronDirection(const double kinE, const double* theGammaDir, double* theDir, RandomEngine* rnge) {
    // -- Sample from SauterGavrila angular distribution, code taken from Geant4:
    // Initial algorithm according Penelope 2008 manual and
    // F.Sauter Ann. Physik 9, 217(1931); 11, 454(1931).
    // Modified according Penelope 2014 manual
    const double tau = kinE*kInvElectronMassC2;
    const double gamma = 1.0 + tau;
    const double beta = std::sqrt(tau*(tau + 2.0))/gamma;

    // ac corresponds to "A" of Eq. (2.31)
    //
    const double ac = (1.0 - beta)/beta;
    const double a1 = 0.5*beta*gamma*tau*(gamma-2.0);
    const double a2 = ac + 2.0;
    // gtmax = maximum of the rejection function according to Eq. (2.28),
    // obtained for tsam=0
    const double gtmax = 2.0*(a1 + 1.0/ac);

    double tsam = 0.0;
    double gtr  = 0.0;

    //2) sampling. Eq. (2.31) of Penelope Manual
    // tsam = 1-std::cos(theta)
    // gtr = rejection function according to Eq. (2.28)
    double rndm[2];
    do {
      rnge->flatArray(2, rndm);
      tsam = 2.0*ac * (2.0*rndm[0] + a2*std::sqrt(rndm[0])) / (a2*a2 - 4.0*rndm[0]);
      gtr = (2.0 - tsam) * (a1 + 1.0/(ac+tsam));
      // Loop checking, 03-Aug-2015, Vladimir Ivanchenko
    } while(rndm[1]*gtmax > gtr);

    const double costheta = 1.0 - tsam;

    const double sint = std::sqrt(tsam*(2.0 - tsam));
    const double phi  = CLHEP::twopi*rnge->flat();

    theDir[0] = sint * std::cos(phi);
    theDir[1] = sint * std::sin(phi);
    theDir[2] = costheta;
    RotateToReferenceFrame(theDir, theGammaDir);
  }
};

#endif // G4HepEmGammaInteractionPhotoelectric_HH
