#ifndef G4HepEmElectronInteractionUMSC_HH
#define G4HepEmElectronInteractionUMSC_HH

#include "G4HepEmConstants.hh"
#include "G4HepEmData.hh"
#include "G4HepEmMacros.hh"
#include "G4HepEmMaterialData.hh"
#include "G4HepEmMath.hh"
#include "G4HepEmMSCTrackData.hh"
#include "G4HepEmParameters.hh"

/**
 * @file    G4HepEmElectronInteractionUMSC.hh
 * @class   G4HepEmElectronInteractionUMSC
 * @author  M. Novak
 * @date    2022
 *
 * @brief Urban model for multiple scattering of e-/e+ for HEP applications.
 */
class G4HepEmElectronInteractionUMSC {
private:
  G4HepEmElectronInteractionUMSC() = delete;

public:

  // The msc step limit will be written into the G4HepEmMSCTrackData::fTrueStepLength member of
  // of the input track. If msc limits the step, this is shorter than the track->GetPStepLength.
  // Note, that in all cases, the final physical step length will need to be coverted to geometrical
  // one that is done in the G4HepEmElectronManager.
  template <typename RandomEngine>
  G4HepEmHostDevice
  static void StepLimit(G4HepEmData* hepEmData, G4HepEmParameters* hepEmPars, G4HepEmMSCTrackData* mscData,
                        double ekin, int imat, double range, double presafety,
                        bool onBoundary, bool iselectron, RandomEngine* rnge) {
    // Initial values:
    //  - lengths are already initialised to the current minimum physics step  which is the true, minimum
    //    step length from all other physics
    //  - the first transport mean free path value, i.e. `mscData.fLambtr1` is also assumed to be up to date
    mscData->fIsNoScatteringInMSC = false;
    mscData->fIsDisplace          = true;

    // stop in case of:
    // - very small steps
    // - will never leave the current volume boundary (the correction on range accounts fluctuation)
    // and indicate no-dispalcement.
    const double kTLimitMinfix = 1.0E-8; // 0.01 [nm] 1.0E-8 [mm]
    const struct G4HepEmMatData& matData = hepEmData->fTheMaterialData->fMaterialData[imat];
    if (mscData->fTrueStepLength < kTLimitMinfix || range*(matData.fUMSCPar) < presafety) {
      mscData->fIsDisplace = false;
      return;
    }
    // set the initial range, dynamic range factor, minimal step and true step values
    // if just enetring to a new volume or performing the very first step with this
    // particle (or more exactly, reaching this point first time)
    if (mscData->fIsFirstStep || onBoundary) {
      const double lambdaTr1 = mscData->fLambtr1;
      mscData->fInitialRange = G4HepEmMax(range, lambdaTr1);
      // note: the below is true only because `lambdaLimit = 1.0 [mm]`
      //const double kILambdaLimit   = 1.0; // 1/(kLambdaLimit = 1.0 [mm])
      mscData->fDynamicRangeFactor = lambdaTr1 > 1.0
                                         ? hepEmPars->fMSCRangeFactor*(0.75 + 0.25*lambdaTr1)
                                         : hepEmPars->fMSCRangeFactor;
      // note: `ekin` below is the kinetic energy in MeV;
      const double stepMin = lambdaTr1*1.0E-3/(2.0E-3 + ekin*(matData.fUMSCStepMinPars[0] + ekin*matData.fUMSCStepMinPars[1]));
      const double dum0    = iselectron ? 0.87*matData.fZeff23 : 0.70*matData.fZeffSqrt;
      // note `tlow` = 5 [keV] ==> 5.0E-3 [MeV] ==> 1/tlow = 200 [1/MeV]
      const double dum1    = ekin > 5.0E-3 ? dum0*stepMin : dum0*stepMin*0.5*(1.0 + ekin*200.0);
      mscData->fTlimitMin  = G4HepEmMax(dum1, kTLimitMinfix);
      // reset first step flag (if any)
      mscData->fIsFirstStep = false;
    }
    // the true step limit
    const double tlimitmin = mscData->fTlimitMin;
    const double tlimit    = range > presafety
                                 ? G4HepEmMax(G4HepEmMax(mscData->fInitialRange*mscData->fDynamicRangeFactor, hepEmPars->fMSCSafetyFactor*presafety), tlimitmin)
                                 : G4HepEmMax(range, tlimitmin);
    // randomise the true step limit but only if the step was determined by msc
    if (tlimit < mscData->fTrueStepLength) { // keep mscData->fTrueStepLength otherwise as teh current step-limit --> not msc limited this step
      const double dum0 = tlimit > tlimitmin
                                     ? G4HepEmMax(rnge->Gauss(tlimit, 0.1*(tlimit - tlimitmin)), tlimitmin)
                                     : tlimitmin;
      mscData->fTrueStepLength = G4HepEmMin(dum0, mscData->fTrueStepLength);
    }
    // msc step limit is done!
    //
    // convert the true (physics) step length to geometrical one (i.e. the pojection of the transport vector
    // along the original direction)
    // NOTE: this conversion is done in the G4HepEmElectronManager
    // ConvertTrueToGeometricLength(...)
  }

  template <typename RandomEngine>
  G4HepEmHostDevice
  static void SampleScattering(G4HepEmData* hepEmData, G4HepEmMSCTrackData* mscData, double pStepLength,
                               double preStepEkin, double preStepTr1mfp, double postStepEkin, double postStepTr1mfp,
                               int imat, bool isElectron, RandomEngine* rnge) {
    const struct G4HepEmMatData& matData = hepEmData->fTheMaterialData->fMaterialData[imat];
    const double cost  = SampleCosineTheta(pStepLength, preStepEkin, preStepTr1mfp, postStepEkin, postStepTr1mfp, mscData->fTlimitMin,
                                                             matData.fRadiationLength, matData.fZeff, matData.fUMSCTailCoeff, matData.fUMSCThetaCoeff, isElectron, rnge);
    // no scattering so no dispacement in case cost = 1.0
    if (std::abs(cost) >= 1.0) {
      mscData->fIsNoScatteringInMSC = true;
      // mscData->fIsDisplace = false;
      return;
    }
    const double sth = std::sqrt((1.0 - cost)*(1.0 + cost));
    const double phi = k2Pi*rnge->flat();
    mscData->SetNewDirection(sth*std::cos(phi), sth*std::sin(phi), cost);
    // compute/sample dispacement
    if (mscData->fIsDisplace && pStepLength > mscData->fZPathLength) {
      SampleDisplacement(pStepLength, phi, mscData, rnge);
    }
  }

  // auxilary method for sampling Urban MSC cos(theta) in the given step (used in the above `SampleScattering`)
  template <typename RandomEngine>
  G4HepEmHostDevice
  static double SampleCosineTheta(double pStepLength, double preStepEkin, double preStepTr1mfp,
                                  double postStepEkin, double postStepTr1mfp, double umscTlimitMin,
                                  double radLength, double zeff, const double* umscTailCoeff, const double* umscThetaCoeff,
                                  bool isElectron, RandomEngine* rnge) {
    // NOTE: since I already know the finalEnergy in the electron Manager and that is already set to the track,
    //       I could get the logFinalEnergy from the updated track which could save up one log call
    // compute the 1rst transport mfp at the post-step point energy
    const double iPreStepTr1mfp = 1.0/preStepTr1mfp;
    const double deltaR1mfp     = preStepTr1mfp - postStepTr1mfp;
    const double tau = std::abs(deltaR1mfp) > 0.01*preStepTr1mfp
                                      ? pStepLength*G4HepEmLog(preStepTr1mfp/postStepTr1mfp)/deltaR1mfp
                                      : pStepLength*iPreStepTr1mfp;
    //
    // Note:  `currentTau = tau` that is used in G4Urban before the displacement
    //        sampling and dispacement is done only if (currentTau >= tausmall)
    //        However, here if (tau < tausmall) we return `cost = 1` that will
    //        result an immediate stop in the caller SampleScattering since this
    //        means no scattering. So we never even try to sample dispacement if
    //        currentTau = tau < tausmall.

    //
    // isotropic case i.e. uniform cost
    const double kTauBig   = 8.0;
    if (tau > kTauBig) {
      return 2.0*rnge->flat() - 1.0;
    }
    //
    // zero scattering i.e. cost = 1.0
    const double kTauSmall = 1.0E-16;
    if (tau < kTauSmall) {
      return 1.0;
    }
    //
    // normal case: sample cost
    double xmeanth, x2meanth;
    if (tau < 0.01) {
      // taylor series approximation
      xmeanth  = 1.0 - tau*(1.0 - 0.5*tau);
      x2meanth = 1.0 - tau*(5.0 - 6.25*tau)*0.333333;
    } else {
      xmeanth  = G4HepEmExp(-tau);
      x2meanth = (1.0 + 2.0*G4HepEmExp(-2.5*tau))*0.333333;
    }
    //  - too large step of low-energy particle
    if (postStepEkin < 0.5*preStepEkin) {
      return SimpleScattering(xmeanth, x2meanth, rnge);
    }
    // - sample theta0 and check if the step is extreme small
    //   note: below `lambdaLimit = 1.0 [mm]`
    const double   tsmall    = G4HepEmMin(umscTlimitMin, 1.0);
    const bool stpNotExSmall = pStepLength > tsmall;
    const double theta0      = stpNotExSmall
                                   ? ComputeTheta0(pStepLength/radLength, postStepEkin, preStepEkin, zeff, umscThetaCoeff, isElectron)
                                   : ComputeTheta0(tsmall/radLength,      postStepEkin, preStepEkin, zeff, umscThetaCoeff, isElectron)*std::sqrt(pStepLength/tsmall);
    //
    if (theta0 > kPi*0.166666) {
      return SimpleScattering(xmeanth, x2meanth, rnge);
    }
    // protection for very small angles (cost = 1)
    const double theta2 = theta0*theta0;
    if (theta2 < kTauSmall) {
      // zero scattering i.e. cost = 1.0
      return 1.0;
    }

    // parameter for tail
    const double dumtau = stpNotExSmall ? tau : tsmall*iPreStepTr1mfp;
    const double parU   = G4HepEmPow(dumtau, 0.1666666);
    const double dumxsi = umscTailCoeff[0] + parU*(umscTailCoeff[1] + parU*umscTailCoeff[2])
                          + umscTailCoeff[3]*G4HepEmLog(pStepLength/(tau*radLength));
    // tail should not be too big
    const double parXsi = G4HepEmMax(dumxsi, 1.9);
    //
    const double   parC =   std::abs(parXsi - 3.) < 0.001 ? 3.001
                          : std::abs(parXsi - 2.) < 0.001 ? 2.001
                                                          : parXsi;
    const double dumC1  = parC - 1.;
    const double dumEa  = G4HepEmExp(-parXsi);
    const double dumEaa = 1./(1. - dumEa);
    double thex = theta2*(1.0 - theta2*0.0833333);
    if (theta2 > 0.01) {
      const double  dum = 2.0*std::sin(0.5*theta0);
      thex = dum*dum;
    }
    const double xmean1 = 1. - (1. - (1. + parXsi)*dumEa)*thex*dumEaa;
    //
    if (xmean1 <= 0.999*xmeanth) {
      return SimpleScattering(xmeanth, x2meanth, rnge);
    }
    //
    const double  x0 = 1. - parXsi*thex;
    const double  bx = parC*thex;

    const double   b = bx + x0;
    const double  b1 = b + 1.;

    const double eb1 = G4HepEmPow(b1, dumC1);
    const double ebx = G4HepEmPow(bx, dumC1);
    const double   d = ebx/eb1;

    const double xmean2 = (x0 + d - (bx - b1*d)/(parC - 2.))/(1. - d);

    const double f1x0 = dumEa*dumEaa;
    const double f2x0 = dumC1/(parC*(1. - d));
    const double prob = f2x0/(f1x0 + f2x0);

    const double qprb = xmeanth/(prob*xmean1 + (1. - prob)*xmean2);
    //
    // sampling of cost
    double rndArray[3];
    rnge->flatArray(3, rndArray);
    if (rndArray[0] < qprb) {
      if (rndArray[1] < prob) {
        return  1. + G4HepEmLog(dumEa + rndArray[2]/dumEaa)*thex;
      } else {
        const double var0 = (1.0 - d)*rndArray[2];
        if (var0 < 0.01*d) {
          const double var = var0/(d*dumC1);
          return -1.0 + var*(1.0 - var*0.5*parC)*b1;
        } else {
          return  1.0 + thex*(parC - parXsi - parC*G4HepEmPow(var0 + d, -1./dumC1));
        }
      }
    } else {
      return 2.0*rndArray[1] - 1.0;
    }
  }

  // auxilary method for sampling cos(theta) in a simplified way: using an arbitrary pdf with correct mean and stdev
  // (used in the above `SampleCosineTheta`)
  template <typename RandomEngine>
  G4HepEmHostDevice
  static double SimpleScattering(double xmeanth, double x2meanth, RandomEngine* rnge) {
    // 'large angle scattering'
    // 2 model functions with correct xmean and x2mean
    const double dum0 = 3.*x2meanth - 1;
    const double dum1 = 2.*xmeanth  - dum0;
    const double    a = 1. + 4.*dum0/dum1;
    const double prob = (2. + a)*xmeanth/a;
    // sampling
    double rndArray[2];
    rnge->flatArray(2, rndArray);
    return (rndArray[0] < prob)
               ? -1. + 2.*G4HepEmPow(rndArray[1],1./(1. + a))
               : -1. + 2.*rndArray[1];
  }

  // auxilary method for computing theta0 (used in the above `SampleCosineTheta`)
  // totry: all these could probably computed in `float`
  G4HepEmHostDevice
  static double ComputeTheta0(double stepInRadLength, double postStepEkin, double preStepEkin,
                              double zeff, const double* umscThetaCoeff, bool isElectron) {
    // ( Highland formula: Particle Physics Booklet, July 2002, eq. 26.10)
    const double     kHighland = 13.6; // note:: assumed to be in MeV
    const double postInvBetaPc = (postStepEkin + kElectronMassC2)/(postStepEkin*(postStepEkin + 2.*kElectronMassC2));
    const double invBetaPc     = preStepEkin != postStepEkin
                                     ? std::sqrt(postInvBetaPc*(preStepEkin + kElectronMassC2)/(preStepEkin*(preStepEkin + 2.*kElectronMassC2)))
                                     : postInvBetaPc;
    const double y = isElectron
                                     ? stepInRadLength
                                     : stepInRadLength * Theta0PositronCorrection(preStepEkin*postStepEkin, zeff);
    return kHighland*std::sqrt(y)*invBetaPc*(umscThetaCoeff[0] + umscThetaCoeff[1]*G4HepEmLog(y));
  }

  // auxilary method for computing the e+ correction to theta0 (used in the above `ComputeTheta0` but only in case of e+)
  G4HepEmHostDevice
  static double Theta0PositronCorrection(double eekin, double zeff);

  // auxilary method for sampling the lateral displacement vector (x,y,0) on a rather approximate way
  // note: should only be called if tru-step-length != z-step-length
  template <typename RandomEngine>
  G4HepEmHostDevice
  static void SampleDisplacement(double pStepLength, double thePhi, G4HepEmMSCTrackData* mscData, RandomEngine* rnge) {
    // simple and fast sampling
    // based on single scattering results
    // u = r/rmax : mean value (r = 0.73*rmax where rmax:= sqrt(t-g)*(t+g))
    const double r = 0.73*std::sqrt((pStepLength - mscData->fZPathLength)*(pStepLength + mscData->fZPathLength));
    // simple distribution for v=Phi-phi=psi ~exp(-beta*v)
    // beta determined from the requirement that distribution should give
    // the same mean value than that obtained from the ss simulation
    const double cbeta  = 2.16;
    const double cbeta1 = 1. - G4HepEmExp(-cbeta*kPi);
    double rndArray[2];
    rnge->flatArray(2, rndArray);
    const double psi = -G4HepEmLog(1. - rndArray[0]*cbeta1)/cbeta;
    const double phi = (rndArray[1] < 0.5) ? thePhi + psi : thePhi - psi;
    mscData->SetDisplacement(r*std::cos(phi), r*std::sin(phi), 0.0);
  }
};

#endif // G4HepEmElectronInteractionUMSC_HH
