
#ifndef G4HepEmGammaInteractionConversion_HH
#define G4HepEmGammaInteractionConversion_HH

#include "G4HepEmData.hh"
#include "G4HepEmElementData.hh"
#include "G4HepEmInteractionUtils.hh"
#include "G4HepEmMacros.hh"
#include "G4HepEmMatCutData.hh"
#include "G4HepEmMaterialData.hh"
#include "G4HepEmRunUtils.hh"

#include <cmath>

class  G4HepEmTLData;

class G4HepEmGammaInteractionConversion {
private:
  G4HepEmGammaInteractionConversion() = delete;

public:
  static void Perform(G4HepEmTLData* tlData, struct G4HepEmData* hepEmData);

  template <typename RandomEngine>
  G4HepEmHostDevice
  static void SampleKinEnergies(struct G4HepEmData* hepEmData, double thePrimEkin, double theLogEkin,
                                int theMCIndx, double& eKinEnergy, double& pKinEnergy, RandomEngine* rnge) {
        // get the material data
        const int               matIndx = (hepEmData->fTheMatCutData->fMatCutData[theMCIndx]).fHepEmMatIndex;
        const G4HepEmMatData&  theMData = hepEmData->fTheMaterialData->fMaterialData[matIndx];
        // sample target element
        const int  elemIndx = (theMData.fNumOfElement > 1)
                                  ? SelectTargetAtom(hepEmData->fTheGammaData, matIndx, thePrimEkin, theLogEkin, rnge->flat())
                                  : 0;
        const int      iZet = theMData.fElementVect[elemIndx];
        const double lpmEnr = kLPMconstant * theMData.fRadiationLength;

        // get the corresponding element data
        const G4HepEmElemData& theElemData = hepEmData->fTheElementData->fElementData[G4HepEmMin(iZet, hepEmData->fTheElementData->fMaxZet)];
        //
        // == Sampling of the total energy, transferred to one of the e-/e+ pair in
        //    units of initial photon energy
        const double eps0 = kElectronMassC2/thePrimEkin;
        double eps = 0.0;
        if (thePrimEkin < 2.0) {
          // uniform sampling at low energies (flat DCS) between the kinematical limits
          // of eps0=mc^2/Eg <= eps <= 0.5 ( symmetric DCS around eps = 0.5)
          eps = eps0 + (0.5-eps0)*rnge->flat();
        } else {
          // use a gamma energy limit of 50.0 [MeV] to turn off Coulomb correction below
          const double deltaFactor = eps0*136./theElemData.fZet13;
          const double    deltaMin = 4.*deltaFactor;
          const double    deltaMax = (thePrimEkin < 50.0) ? theElemData.fDeltaMaxLow : theElemData.fDeltaMaxHigh;
          const double    logZ13   = 0.333333*theElemData.fLogZ;
          const double          FZ = (thePrimEkin < 50.0) ? 8.*logZ13 : 8.*(logZ13 + theElemData.fCoulomb);
          // compute the limits of eps
          const double        epsp = 0.5 - 0.5*std::sqrt(1. - deltaMin/deltaMax) ;
          const double      epsMin = G4HepEmMax(eps0, epsp);
          const double    epsRange = 0.5 - epsMin;
          //
          // sample the energy rate (eps) of the created electron (or positron)
          double F10, F20;
          ScreenFunction12(deltaMin, F10, F20);
          F10 -= FZ;
          F20 -= FZ;
          const double NormF1   = G4HepEmMax(F10 * epsRange * epsRange, 0.);
          const double NormF2   = G4HepEmMax(1.5 * F20                , 0.);
          const double NormCond = NormF1/(NormF1 + NormF2);
          // check if LPM correction is active ( active if gamma energy > 100 [GeV])
          eps = (thePrimEkin < 100000.0)
                    ? SampleEnergyRateNoLPM  (NormCond, epsMin, epsRange, deltaFactor, 1./F10, 1./F20, FZ, rnge)
                    : SampleEnergyRateWithLPM(NormCond, epsMin, epsRange, deltaFactor, 1./F10, 1./F20, FZ, rnge,
                                              thePrimEkin, lpmEnr, &theElemData);
        }
        //
        // select charges randomly and compute kinetic
        double eTotEnergy, pTotEnergy;
        if (rnge->flat() > 0.5) {
          eTotEnergy = (1.-eps)*thePrimEkin;
          pTotEnergy = eps*thePrimEkin;
        } else {
          pTotEnergy = (1.-eps)*thePrimEkin;
          eTotEnergy = eps*thePrimEkin;
        }
        //
        // compute kinetic energies of the e- and e+ pair.
        eKinEnergy = G4HepEmMax(0.,eTotEnergy - kElectronMassC2);
        pKinEnergy = G4HepEmMax(0.,pTotEnergy - kElectronMassC2);
      }


  template <typename RandomEngine>
  G4HepEmHostDevice
  static void SampleDirections(const double* orgGammaDir, double* secElDir, double* secPosDir,
                               const double secElEkin, const double secPosEkin, RandomEngine* rnge) {
    // sample azimuthal angle (2Pi symmetric)
    const double  phi    = k2Pi*rnge->flat();
    const double cosPhi  = std::cos(phi);
    const double sinPhi  = std::sin(phi);
    // sample e- cos(theta) by using the (modified Tsai sampling:
    const double costEl  = SampleCostModifiedTsai(secElEkin, rnge);
    const double sintEl  = std::sqrt((1.0-costEl)*(1.0+costEl));
    secElDir[0] = sintEl * cosPhi;
    secElDir[1] = sintEl * sinPhi;
    secElDir[2] = costEl;
    // rotate to refernce frame (G4HepEmRunUtils function) to get it in lab. frame
    RotateToReferenceFrame(secElDir, orgGammaDir);
    // sample e+ cos(theta) by using the (modified Tsai sampling:
    const double costPos = SampleCostModifiedTsai(secPosEkin, rnge);
    const double sintPos  = std::sqrt((1.0-costPos)*(1.0+costPos));
    secPosDir[0] = -sintPos * cosPhi;
    secPosDir[1] = -sintPos * sinPhi;
    secPosDir[2] = costPos;
    // rotate to refernce frame (G4HepEmRunUtils function) to get it in lab. frame
    RotateToReferenceFrame(secPosDir, orgGammaDir);
  }


  // Target atom selector for the above bremsstrahlung intercations in case of
  // materials composed from multiple elements.
  G4HepEmHostDevice
  static int SelectTargetAtom(const struct G4HepEmGammaData* gmData, const int imat, const double ekin,
                              const double lekin, const double urndn);

  template <typename RandomEngine>
  G4HepEmHostDevice
  static double SampleEnergyRateNoLPM(const double normCond, const double epsMin, const double epsRange,
                                      const double deltaFactor, const double invF10, const double invF20,
                                      const double fz, RandomEngine* rnge) {
    double rndmv[3];
    double greject = 0.;
    double eps     = 0.;
    do {
      rnge->flatArray(3, rndmv);
      if (normCond > rndmv[0]) {
        eps = 0.5 - epsRange * std::pow(rndmv[1], 1./3.);//G4HepEmX13(rndmv[1]);
        const double delta = deltaFactor/(eps*(1.-eps));
        greject = (ScreenFunction1(delta)-fz)*invF10;
      } else {
        eps = epsMin + epsRange*rndmv[1];
        const double delta = deltaFactor/(eps*(1.-eps));
        greject = (ScreenFunction2(delta)-fz)*invF20;
      }
      // Loop checking, 03-Aug-2015, Vladimir Ivanchenko
    } while (greject < rndmv[2]);
    //  end of eps sampling
    return eps;
  }

  template <typename RandomEngine>
  G4HepEmHostDevice
  static double SampleEnergyRateWithLPM(const double normCond, const double epsMin, const double epsRange,
                                        const double deltaFactor, const double invF10, const double invF20,
                                        const double fz, RandomEngine* rnge, const double eGamma,
                                        const double lpmEnergy, const struct G4HepEmElemData* elemData) {
    const double         z23 = elemData->fZet23;
    const double     ilVarS1 = elemData->fILVarS1;
    const double ilVarS1Cond = elemData->fILVarS1Cond;
    double rndmv[3];
    double greject = 0.;
    double eps     = 0.;
    do {
      rnge->flatArray(3, rndmv);
      if (normCond > rndmv[0]) {
        eps = 0.5 - epsRange * std::pow(rndmv[1], 1./3.); //G4HepEmX13(rndmv[1]);
        const double delta = deltaFactor/(eps*(1.-eps));
        double funcXiS, funcGS, funcPhiS, phi1, phi2;
        ComputePhi12(delta, phi1, phi2);
        //  0.0 = no density effect correction (only in case of Brem.)
        // +1.0 => Brem:  s' = sqrt{ 0.125 E_lpm E_g / [ E_t ( E_t - E_g) ]  }
        // -1.0 => Pair:  s' = sqrt{ 0.125 E_lpm E_g / [ E_t ( E_g - E_t) ]  } with E_t = eps*E_g
        EvaluateLPMFunctions(funcXiS, funcGS, funcPhiS, eGamma, eps*eGamma, lpmEnergy, z23, ilVarS1, ilVarS1Cond, 0.0, -1.0);
        greject = funcXiS*((2.*funcPhiS+funcGS)*phi1-funcGS*phi2-funcPhiS*fz)*invF10;
      } else {
        eps = epsMin + epsRange*rndmv[1];
        const double delta = deltaFactor/(eps*(1.-eps));
        double funcXiS, funcGS, funcPhiS, phi1, phi2;
        ComputePhi12(delta, phi1, phi2);
        EvaluateLPMFunctions(funcXiS, funcGS, funcPhiS, eGamma, eps*eGamma, lpmEnergy, z23, ilVarS1, ilVarS1Cond, 0.0, -1.0);
        greject = funcXiS*( (funcPhiS+0.5*funcGS)*phi1 + 0.5*funcGS*phi2
                             -0.5*(funcGS+funcPhiS)*fz)*invF20;
      }
    } while (greject < rndmv[2]);
    //  end of eps sampling
    return eps;
  }

  G4HepEmHostDevice
  static void ComputePhi12(const double delta, double &phi1, double &phi2);


  // Compute the value of the screening function 3*PHI1(delta) - PHI2(delta):
  G4HepEmHostDevice
  static double ScreenFunction1(const double delta);

  // Compute the value of the screening function 1.5*PHI1(delta) +0.5*PHI2(delta):
  G4HepEmHostDevice
  static double ScreenFunction2(const double delta);

  // Same as ScreenFunction1 and ScreenFunction2 but computes them at once
  G4HepEmHostDevice
  static void ScreenFunction12(const double delta, double &f1, double &f2);
};

#endif  // G4HepEmGammaInteractionConversion_HH
