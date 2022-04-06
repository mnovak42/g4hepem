
#ifndef G4HepEmElectronManager_HH
#define G4HepEmElectronManager_HH

#include "G4HepEmConstants.hh"
#include "G4HepEmData.hh"
#include "G4HepEmElectronEnergyLossFluctuation.hh"
#include "G4HepEmElectronInteractionUMSC.hh"
#include "G4HepEmElectronTrack.hh"
#include "G4HepEmMacros.hh"
#include "G4HepEmMatCutData.hh"
#include "G4HepEmMaterialData.hh"
#include "G4HepEmParameters.hh"
#include "G4HepEmRunUtils.hh"
#include "G4HepEmTrack.hh"

struct G4HepEmElectronData;
class  G4HepEmTLData;
class  G4HepEmMSCTrackData;

/**
 * @file    G4HepEmElectronManager.hh
 * @struct  G4HepEmElectronManager
 * @author  M. Novak
 * @date    2020
 *
 * @brief The top level run-time manager for e-/e+ transport simulations.
 *
 * This manager can provide the information regarding how far a given e-/e+ particle
 * goes along its original direction till it's needed to be stopped again because
 * some physics interaction(s) needs to be performed. It is also responsible to
 * perform the required interaction(s) as well.
 *
 * The two methods, through wich this manager acts on the particles, are the
 * G4HepEmElectronManager::HowFar() and G4HepEmElectronManager::Perform(). The
 * first provides the information regarding how far the particle can go, along its
 * original direction, till its next stop due to physics interaction(s).
 * The second can be used to perform the corresponding physics interaction(s).
 * All physics interactions, relevant for HEP detector simulatios, such as
 * `ionisation`, `bremsstrahlung`, `Coulomb scattering` are considered for e-/e+
 * with `annihilation` in addition for e+, including both their continuous, discrete
 * and at-rest parts pespectively. The accuracy of the models, used to describe
 * these interactions, are also compatible to those used by HEP detector simulations.
 *
 * Each G4HepEmRunManager has its own member from this manager for e-/e+ transport.
 * However, a single object could alos be used and shared by all the worker run
 * managers since this G4HepEmElectronManager is stateless. All the state and
 * thread related infomation (e.g. primary/secondary tracks or the thread local
 * random engine) are stored in the G4HepEmTLData input argument, that is also
 * used to deliver the effect of the actions of this manager (i.e. written into
 * the tracks stored in the input G4HepEmTLData argument).
 */

class G4HepEmElectronManager {
private:
  G4HepEmElectronManager() = delete;

public:

  /** Functions that provides the information regarding how far a given e-/e+ particle goes.
    *
    * This functions provides the information regarding how far a given e-/e+ particle goes
    * till it's needed to be stopped again because some physics interaction(s) needs to be performed.
    * The input/primary e-/e+ particle track is provided through the G4HepEmTLData input argument. The
    * The computed physics step lenght is written directly into the input track. There is no any local
    * (state) variable used in the computation.
    *
    * @param hepEmData pointer to the top level, global, G4HepEmData structure.
    * @param hepEmPars pointer to the global, G4HepEmParameters structure.
    * @param tlData    pointer to a worker-local, G4HepEmTLData object. The corresonding object
    *   is assumed to contain all the required input information in its primary G4HepEmTLData::fElectronTrack
    *   member. This member is also used to deliver the results of the function call, i.e. the computed physics
    *   step limit is written into the G4HepEmTLData::fElectronTrack (in its fGStepLength member).
    */
  static void HowFar(struct G4HepEmData* hepEmData, struct G4HepEmParameters* hepEmPars, G4HepEmTLData* tlData);

  /** Function that provides the information regarding how far a given e-/e+ particle goes.
    *
    * This function provides the information regarding how far a given e-/e+ particle goes
    * till it's needed to be stopped again because some physics interaction(s) needs to be performed.
    * The input/primary e-/e+ particle track is provided as G4HepEmElectronTrack which must have sampled
    * `number-of-interaction-left`. The computed physics step length is written directly into the input
    * track. There is no local (state) variable used in the computation.
    *
    * @param hepEmData pointer to the top level, global, G4HepEmData structure.
    * @param hepEmPars pointer to the global, G4HepEmParameters structure.
    * @param theElTrack pointer to the input information of the track. The data structure must have all entries
    *   `number-of-interaction-left` sampled and is also used to deliver the results of the function call, i.e.
    *   the computed physics step limit is written into its fGStepLength member.
    */
  template <typename RandomEngine>
  G4HepEmHostDevice
  static void HowFar(struct G4HepEmData* hepEmData, struct G4HepEmParameters* hepEmPars, G4HepEmElectronTrack* theElTrack, RandomEngine* rnge) {
    int indxWinnerProcess = -1;  // init to continous
    // === 1. Continuous energy loss limit
    double pStepLength     = kALargeValue;
    G4HepEmTrack* theTrack = theElTrack->GetTrack();
    const double   theEkin = theTrack->GetEKin();
    const double  theLEkin = theTrack->GetLogEKin();
    const int       theIMC = theTrack->GetMCIndex();
    const bool  isElectron = (theTrack->GetCharge() < 0.0);

    const G4HepEmElectronData* theElectronData = isElectron
                                                     ? hepEmData->fTheElectronData
                                                     : hepEmData->fThePositronData;
    //
    const double range  = GetRestRange(theElectronData, theIMC, theEkin, theLEkin);
    theElTrack->SetRange(range);
    const double frange = hepEmPars->fFinalRange;
    const double drange = hepEmPars->fDRoverRange;
    pStepLength = (range > frange)
                              ? range*drange + frange*(1.0-drange)*(2.0-frange/range)
                              : range;
    //  std::cout << " pStepLength = " << pStepLength << " range = " << range << " frange = " << frange << std::endl;
    // === 2. Discrete limits due to eestricted Ioni and Brem (accounting e-loss)
    const int theImat = (hepEmData->fTheMatCutData->fMatCutData[theIMC]).fHepEmMatIndex;
    double mxSecs[3];
    // ioni, brem and annihilation to 2 gammas (only for e+)
    mxSecs[0] = GetRestMacXSecForStepping(theElectronData, theIMC, theEkin, theLEkin, true);
    mxSecs[1] = GetRestMacXSecForStepping(theElectronData, theIMC, theEkin, theLEkin, false);
    mxSecs[2] = (isElectron)
                    ? 0.0
                    : ComputeMacXsecAnnihilationForStepping(theEkin, hepEmData->fTheMaterialData->fMaterialData[theImat].fElectronDensity);
    // compute mfp and see if we need to sample the `number-of-interaction-left`
    // before we use it to get the current discrete proposed step length
    for (int ip=0; ip<3; ++ip) {
      const double mxsec = mxSecs[ip];
      const double   mfp = (mxsec>0.) ? 1./mxsec : kALargeValue;
      // save the mac-xsec for the update of the `number-of-interaction-left`:
      // the `number-of-intercation-left` should be updated in the along-step-action
      // after the MSC has changed the step.
      theTrack->SetMFP(mfp, ip);
      // sample the proposed step length
      const double dStepLimit = mfp*theTrack->GetNumIALeft(ip);
      if (dStepLimit<pStepLength) {
        pStepLength = dStepLimit;
        indxWinnerProcess = ip;
      }
    }
    //
    // Now MSC is called to see:
    // - if it limits the (true, i.e. physical) step length further
    // Then we perform the physical --> geometric step length conversion here:
    //  - it provides the projection of the transport vector along the original
    //    direction.
    // Note, this later also part of the MSC model and might also limit the true
    // step length further (see my note inside ConvertTrueToGeometricLength though).
    // Therefore, the check if MSC limited the step must be done after
    // the physical -->  geometric (i.e. true to geom.) conversion.
    //
#ifndef NOMSC
    G4HepEmMSCTrackData* mscData = theElTrack->GetMSCTrackData();
    // init some mscData for the case if we skipp calling msc due to very small step
    mscData->fTrueStepLength      = pStepLength;
    mscData->fZPathLength         = pStepLength;
    mscData->fIsActive            = false;
    mscData->SetDisplacement(0., 0., 0.);
    mscData->SetNewDirection(0., 0., 1.);
    // no msc in case of very small steps
    const double kGeomMinLength = 5.E-8; // 0.05 [nm]
    if (pStepLength > kGeomMinLength && theEkin > 1.0E-3) {
      mscData->fIsActive = true;
      // compute the fist transport mean free path
      mscData->fLambtr1  = GetTransportMFP(theElectronData, theImat, theEkin, theLEkin);
      G4HepEmElectronInteractionUMSC::StepLimit(hepEmData, hepEmPars, mscData, theEkin, theImat, range,
                                                theTrack->GetSafety(), theTrack->GetOnBoundary(), isElectron, rnge);
      // If msc limited the true step length, then the G4HepEmMSCTrackData::fTrueStepLength member of
      // the input electron track is < pStepLengt. Otherwise its = pStepLengt.
      // Call the True --> Geometric conversion since that might limits further the true step Length:
      //   - convert the physical step length to geometrical one. The result will be
      //     written into mscData::fZPathLength.
      ConvertTrueToGeometricLength(hepEmData, mscData, theEkin, range, theIMC, isElectron);
      // check now if msc limited the step:
      const double mscTruStepLength = mscData->fTrueStepLength;
      if (mscTruStepLength < pStepLength) {
        // indicate continuous step limit as msc limited the step and set the new pStepLengt
        indxWinnerProcess = -2;
        pStepLength = mscTruStepLength;
      }
    }
    // set geometrical step length (protect agains wrong conversion, i.e. if gL > pL)
    theTrack->SetGStepLength(G4HepEmMin(mscData->fZPathLength, pStepLength));

    // finally set the true (physical) step length and the winner process index of this primary track
    theElTrack->SetPStepLength(pStepLength);
    theTrack->SetWinnerProcessIndex(indxWinnerProcess);
#else
    theElTrack->SetPStepLength(pStepLength);
    theTrack->SetWinnerProcessIndex(indxWinnerProcess);
    theTrack->SetGStepLength(pStepLength);
#endif
  }

  /** Functions that performs all continuous physics interactions for a given e-/e+ particle.
    *
    * This functions can be invoked when the particle is propagated to its post-step point to perform all
    * continuous physics interactions. The input/primary e-/e+ particle track is provided through as
    * G4HepEmElectronTrack. There is no local (state) variable used in the computation.
    *
    * @param hepEmData pointer to the top level, global, G4HepEmData structure.
    * @param hepEmPars pointer to the global, G4HepEmParameters structure.
    * @param theElTrack pointer to the input information of the track. All the results of this function call,
    *   i.e. the primary particle's energy updated to its post-interaction(s), are also delivered through this
    *   object.
    * @return boolean whether the particle was stopped
    */
  // Here I can have my own transportation to be called BUT at the moment I cannot
  // skip the G4Transportation if I do it by myself !!!

  // Note: energy deposit will be set here i.e. this is the first access to it that
  //       will clear the previous step value.
  template <typename RandomEngine>
  G4HepEmHostDevice
  static bool PerformContinuous(struct G4HepEmData* hepEmData, struct G4HepEmParameters* hepEmPars, G4HepEmElectronTrack* theElTrack, RandomEngine* rnge) {
    // === 1. MSC should be invoked to obtain the physics step Length
    G4HepEmTrack*   theTrack = theElTrack->GetTrack();
    // call MSC::ConvertGeometricToTrueLength that will provide the true (i.e. physical)
    // step length in the G4HepEmMSCTrackData::fTrueStepLength member.
    // NOTE: in case the step was NOT limited by boundary, we know the true step length since
    //       the particle went as far as we expected.
    const bool        isElectron = (theTrack->GetCharge() < 0.0);
    const double        theEkin  = theTrack->GetEKin();
    const double        theRange = theElTrack->GetRange();

#ifndef NOMSC
    const double gStepLength = theTrack->GetGStepLength();
    double pStepLength = gStepLength;
    bool isScattering = false;
    G4HepEmMSCTrackData* mscData = theElTrack->GetMSCTrackData();
    if (mscData->fIsActive) {
      pStepLength = mscData->fTrueStepLength;
      isScattering = true;
      // if we hit boundary or stopped before we wanted for any reasons: convert geom. -> true
      if (gStepLength < mscData->fZPathLength) {
        // the converted geom --> true step Length will be written into mscData::fTrueStepLength
        ConvertGeometricToTrueLength(mscData, theRange, gStepLength);
        // protect against wrong true --> geom --> true conevrsion: physical step
        // cannot be longer than before converted to geometrical
        pStepLength = G4HepEmMin(pStepLength, mscData->fTrueStepLength);
        // store the final true step length value
        mscData->fTrueStepLength = pStepLength;
      }
      // optimisation: do not sample msc and dispalcement in case of last (rangeing out) or short steps
      const double kGeomMinLength = 5.E-8; // 0.05 [nm]
      if (pStepLength <= kGeomMinLength || theRange <= pStepLength) {
        isScattering = false;
      }
    }

/*
  if (isScattering) {
    const G4HepEmElectronData* elData0 = isElectron
                                        ? hepEmData->fTheElectronData
                                        : hepEmData->fThePositronData;

    const double postStepRange = theRange - pStepLength;
    double eloss0 = theEkin - GetInvRange(elData0, theTrack->GetMCIndex(), postStepRange);

    G4HepEmElectronInteractionMSC::SampleScattering(hepEmData, mscData, pStepLength, theEkin, eloss0, theTrack->GetMCIndex(), isElectron, rnge);
    // NOTE: displacement will be applied in the caller where we have access to the required Geant4 functionality
    //       (and if its length is longer than a small minimal length and we are not ended up on boundary)
    //
    // rotate direction and displacement vectors (if any) and update new direction of the primary
    if (!(mscData->fIsNoScatteringInMSC)) {
      RotateToReferenceFrame(mscData->fDirection, theTrack->GetDirection());
      if (!(mscData->fIsNoDisplace)) {
        RotateToReferenceFrame(mscData->fDisplacement, theTrack->GetDirection());
      }
      // upadte new direction
      theTrack->SetDirection(mscData->fDirection);
    }
  }
*/
#else
    double pStepLength = theTrack->GetGStepLength();
#endif
    // set the results of the geom ---> true in the primary e- etrack
    theElTrack->SetPStepLength(pStepLength);
    //
    if (pStepLength<=0.0) {
      return false;
    }
    // compute the energy loss first based on the new step length: it will be needed in the
    // MSC scatteirng and displacement computation here as well (that is done only if not
    // the last step with the particle).
    // But update the number of interaction length left before.
    //
    // === 2. The `number-of-interaction-left` needs to be updated based on the actual
    //        physical step Length
    double*    numInterALeft = theTrack->GetNumIALeft();
    double*       preStepMFP = theTrack->GetMFP();
    numInterALeft[0] -= pStepLength/preStepMFP[0];
    numInterALeft[1] -= pStepLength/preStepMFP[1];
    numInterALeft[2] -= pStepLength/preStepMFP[2];
    //
    // === 3. Continuous energy loss needs to be computed
    // 3./1. stop tracking when reached the end (i.e. it has been ranged out by the limit)
    // @TODO: actually the tracking cut is around 1 keV and the min-table energy is 100 eV so the second should never
    //        under standard EM constructor configurations
    if (pStepLength >= theRange || theEkin <= hepEmPars->fMinLossTableEnergy) {
      // stop and deposit the remaining energy
      theTrack->SetEnergyDeposit(theEkin);
      theTrack->SetEKin(0.0);
      return true;
    }
    // 3/1. try linear energy loss approximation:
    const G4HepEmElectronData* elData = isElectron
                                            ? hepEmData->fTheElectronData
                                            : hepEmData->fThePositronData;
    // NOTE: this is the pre-step IMC !!!
    const int      theIMC = theTrack->GetMCIndex();
    const double theLEkin = theTrack->GetLogEKin();
    double eloss = pStepLength*GetRestDEDX(elData, theIMC, theEkin, theLEkin);
    // 3/2. use integral if linear energy loss is over the limit fraction
    if (eloss > theEkin*hepEmPars->fLinELossLimit) {
      const double postStepRange = theRange - pStepLength;
      eloss = theEkin - GetInvRange(elData, theIMC, postStepRange);
    }
    eloss = G4HepEmMax(eloss, 0.0);
#if !defined(NOMSC) || !defined(NOFLUCTUATION)
    // keep the mean energy loss
    const double meanELoss = eloss;
#endif
#ifndef NOMSC
    bool isActiveEnergyLossFluctuation = false;
#endif
    if (eloss >= theEkin) {
      eloss = theEkin;
    } else {
#ifndef NOFLUCTUATION
      // sample energy loss fluctuations
      const double kFluctParMinEnergy  = 1.E-5; // 10 eV
      if (meanELoss > kFluctParMinEnergy) {
#ifndef NOMSC
        isActiveEnergyLossFluctuation = true;
#endif
        const G4HepEmMCCData& theMatCutData = hepEmData->fTheMatCutData->fMatCutData[theIMC];
        const double elCut   = theMatCutData.fSecElProdCutE;
        const int    theImat = theMatCutData.fHepEmMatIndex;
        const double meanExE = hepEmData->fTheMaterialData->fMaterialData[theImat].fMeanExEnergy;
        //
        const double tmax = isElectron ? 0.5*theEkin : theEkin;
        const double tcut = G4HepEmMin(elCut, tmax);
        eloss = G4HepEmElectronEnergyLossFluctuation::SampleEnergyLossFLuctuation(theEkin, tcut, tmax,
                                                                                              meanExE, pStepLength, meanELoss, rnge);
      }
#endif
    }
    eloss = G4HepEmMax(eloss, 0.0);
    //
    // 3/3. check if final kinetic energy drops below the tracking cut and stop
    double finalEkin = theEkin - eloss;
    if (finalEkin <= hepEmPars->fElectronTrackingCut) {
      eloss     = theEkin;
      finalEkin = 0.0;
      theTrack->SetEKin(finalEkin);
      theTrack->SetEnergyDeposit(eloss);
      return true;
    }
    //
    theTrack->SetEKin(finalEkin);
    theTrack->SetEnergyDeposit(eloss);


#ifndef NOMSC
    //
    // Complete here the MSC part by computing the net angular deflection and dispalcement
    //
    // Smaple scattering in MSC and compute the new direction and displacement vectors (if any)
    // The new direction and dispalcement vectors, proposed by MSC, are given in mscData::fDirection and
    // mscData::fDisplacement.
    const double kTLimitMinfix = 1.0E-8; // 0.01 [nm] 1.0E-8 [mm]
    const double kTauSmall     = 1.0e-16;
    if (isScattering && (pStepLength > G4HepEmMax(kTLimitMinfix, kTauSmall*mscData->fLambtr1))) {
      // only to make sure that we also use E2 = E1 under the same condition as in G4
      double postStepEkin  = theEkin;
      double postStepLEkin = theLEkin;
      if (pStepLength > theRange*0.01) {
        // meanELoss = eloss when the energy loss fluct. is NOT active so postStepEkin = finalEkin in that
        // case which can save a log call for us since we can get the log-finalEkin now from the track
        if (!isActiveEnergyLossFluctuation) {
          postStepEkin  = theTrack->GetEKin();
          postStepLEkin = theTrack->GetLogEKin();
        } else {
          // Otherwise subtract the mean energy loss and compute the logarithm.
          postStepEkin  = theEkin - meanELoss;
          postStepLEkin = G4HepEmLog(postStepEkin);
        }
      }
      // sample msc scattering:
      // - compute the fist transport mean free path at the post-step energy point
      const int           theImat = (hepEmData->fTheMatCutData->fMatCutData[theIMC]).fHepEmMatIndex;
      const double postStepTr1mfp = GetTransportMFP(elData, theImat, postStepEkin, postStepLEkin);
      // - sample scattering: including net angular deflection and lateral dispacement that will be
      //                      written into mscData::fDirection and mscData::fDisplacement
      G4HepEmElectronInteractionUMSC::SampleScattering(hepEmData, mscData, pStepLength, theEkin, mscData->fLambtr1, postStepEkin, postStepTr1mfp,
                                                       theImat, isElectron, rnge);
      // NOTE: displacement will be applied in the caller where we have access to the required Geant4 functionality
      //       (and if its length is longer than a small minimal length and we are not ended up on boundary)
      //
      // rotate direction and displacement vectors (if any) and update new direction of the primary
      if (!(mscData->fIsNoScatteringInMSC)) {
        RotateToReferenceFrame(mscData->fDirection, theTrack->GetDirection());
        if (mscData->fIsDisplace) {
          RotateToReferenceFrame(mscData->fDisplacement, theTrack->GetDirection());
        }
        // upadte new direction
        theTrack->SetDirection(mscData->fDirection);
      }
    }
#endif
    return false;
  }

  /** Function to check if a delta interaction happens instead of the discrete process.
    *
    * @param hepEmData pointer to the top level, global, G4HepEmData structure.
    * @param hepEmPars pointer to the global, G4HepEmParameters structure.
    * @param theTrack pointer to the input information of the track.
    * @param rand number drawn at random
    * @return boolean whether a delta interaction happens
    */
  G4HepEmHostDevice
  static bool CheckDelta(struct G4HepEmData* hepEmData, G4HepEmTrack* theTrack, double rand);

  /** Functions that performs all physics interactions for a given e-/e+ particle.
    *
    * This functions can be invoked when the particle is propagated to its post-step point to perform all
    * physics interactions. The input/primary e-/e+ particle track is provided through the G4HepEmTLData input
    * argument. The post-interaction(s) primary track and the secondary tracks are also provided through this
    * G4HepEmTLData input argument. There is no any local (state) variable used in the computation.
    *
    * @param hepEmData pointer to the top level, global, G4HepEmData structure.
    * @param hepEmPars pointer to the global, G4HepEmParameters structure.
    * @param tlData    pointer to a worker-local, G4HepEmTLData object. The corresonding object
    *   is assumed to contain all the required input information in its primary G4HepEmTLData::fElectronTrack
    *   member. All the results of this function call, i.e. the primary particle updated to its post-interaction(s)
    *   state as well as the possible secondary particles, are also delivered through this G4HepEmTLData.
    */
  static void Perform(struct G4HepEmData* hepEmData, struct G4HepEmParameters* hepEmPars, G4HepEmTLData* tlData);

  /// The following functions are not meant to be called directly by clients, only from tests.

  /**
    * Auxiliary function that evaluates and provides the `restricted range` for the given kinetic energy
    * and material-cuts combination.
    *
    * @param elData pointer to the global e-/e+ data structure that contains the corresponding `Energy Loss` related data.
    * @param imc    index of the ``G4HepEm`` material-cuts in which the range is required
    * @param ekin   kinetic energy of the e-/e+ at which the range is required
    * @param lekin  logarithm of the above kinetic energy
    * @return `Restricted range` value, interpolated at the given e-/e+ kinetic energy in the given material-cuts based on
    *   the corresponding (discrete) `Energy Loss` data provded as input.
    */

  G4HepEmHostDevice
  static double GetRestRange(const struct G4HepEmElectronData* elData, const int imc, const double ekin, const double lekin);

  G4HepEmHostDevice
  static double GetRestDEDX(const struct G4HepEmElectronData* elData, const int imc, const double ekin, const double lekin);

  G4HepEmHostDevice
  static double GetInvRange(const struct G4HepEmElectronData* elData, int imc, double range);

  G4HepEmHostDevice
  static double GetRestMacXSec(const struct G4HepEmElectronData* elData, const int imc, const double ekin,
                               const double lekin, bool isioni);

  G4HepEmHostDevice
  static double GetRestMacXSecForStepping(const struct G4HepEmElectronData* elData, const int imc, double ekin,
                                          double lekin, bool isioni);

  G4HepEmHostDevice
  static double GetTransportMFP(const struct G4HepEmElectronData* elData, const int im, const double ekin, const double lekin);

  G4HepEmHostDevice
  static double ComputeMacXsecAnnihilation(const double ekin, const double electronDensity);

  G4HepEmHostDevice
  static double ComputeMacXsecAnnihilationForStepping(const double ekin, const double electronDensity);

  G4HepEmHostDevice
  static void   ConvertTrueToGeometricLength(const G4HepEmData* hepEmData, G4HepEmMSCTrackData* mscData,
                                             double ekin, double range, int imc, bool iselectron);

  G4HepEmHostDevice
  static void   ConvertGeometricToTrueLength(G4HepEmMSCTrackData* mscData, double range, double gStepToConvert);
};


#endif // G4HepEmElectronManager_HH
