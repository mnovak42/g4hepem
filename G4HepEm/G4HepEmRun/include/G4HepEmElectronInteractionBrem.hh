
#ifndef G4HepEmElectronInteractionBrem_HH
#define G4HepEmElectronInteractionBrem_HH

#include "G4HepEmElementData.hh"
#include "G4HepEmData.hh"
#include "G4HepEmInteractionUtils.hh"
#include "G4HepEmMacros.hh"
#include "G4HepEmMatCutData.hh"
#include "G4HepEmMaterialData.hh"
#include "G4HepEmSBTableData.hh"

class  G4HepEmTLData;
struct G4HepEmElectronData;

// Bremsstrahlung interaction based on:
// 1. SB: - the numerical Seltzer-Berger DCS for the emitted photon energy.
//        - used between 1 keV - 1 GeV primary e-/e+ kinetic energies.
//        NOTE: the core part i.e. sampling the emitted photon energy is different than
//          that in the G4SeltzerBergerModel. I implemented here my rejection free,
//          memory effcicient (tables only per Z and not per mat-cuts) sampling.
//          Rejection is used only to account dielectric supression and e+ correction.
// 2. RB: - the Bethe-Heitler DCS with modifications such as screening and Coulomb
//          corrections, emission in the field of the atomic electrons and LPM suppression.
//          Used between 1 GeV - 100 TeV primary e-/e+ kinetic energies.
class G4HepEmElectronInteractionBrem {
private:
  G4HepEmElectronInteractionBrem() = delete;

public:
  static void Perform(G4HepEmTLData* tlData, struct G4HepEmData* hepEmData, bool iselectron, bool isSBmodel);


  // Sampling of the energy transferred to the emitted photon using the numerical
  // Seltzer-Berger DCS.
  template <typename RandomEngine>
  G4HepEmHostDevice
  static double SampleETransferSB(struct G4HepEmData* hepEmData, double thePrimEkin, double theLogEkin,
                                  int theMCIndx, RandomEngine* rnge, bool iselectron) {
    const G4HepEmMCCData& theMCData = hepEmData->fTheMatCutData->fMatCutData[theMCIndx];
    const double          theGamCut = theMCData.fSecGamProdCutE;
    const double       theLogGamCut = theMCData.fLogSecGamCutE;
    const G4HepEmMatData& theMData = hepEmData->fTheMaterialData->fMaterialData[theMCData.fHepEmMatIndex];
    // sample target element
    const G4HepEmElectronData* theElData = iselectron ? hepEmData->fTheElectronData : hepEmData->fThePositronData;
    const int elemIndx = (theMData.fNumOfElement > 1)
                                               ? SelectTargetAtom(theElData, theMCIndx, thePrimEkin, theLogEkin, rnge->flat(), true)
                                               : 0;
    const int     iZet = theMData.fElementVect[elemIndx];
    const double  dZet = (double)iZet;
    //
    // == Sampling of the emitted photon energy
    // get the G4HepEmSBTableData structure
    const G4HepEmSBTableData* theSBTables = hepEmData->fTheSBTableData;
    // get the start index of sampling tables for this Z
    const int iStart   = theSBTables->fSBTablesStartPerZ[iZet];
    // get the index of the gamma-cut cumulative in this Z data that corresponds to this mc
    const int iGamCut  = theSBTables->fGammaCutIndices[theSBTables->fGammaCutIndxStartIndexPerMC[theMCIndx]+elemIndx];
    // find the lower energy grid index i.e. `i` such that E_i <= E < E_{i+1}
    // find lower e- energy bin
    bool      isCorner = false; // indicate that the lower edge e- energy < gam-gut
    bool      isSimply = false; // simply sampling: isCorner+lower egde is selected
    int   elEnergyIndx = (int)(theSBTables->fSBTableData[iStart+2]);  // maxE-grid index for this Z
    // only if e- ekin is below the maximum value(use table at maximum otherwise)
    if (thePrimEkin < theSBTables->fElEnergyVect[elEnergyIndx]) {
      const double val = (theLogEkin-theSBTables->fLogMinElEnergy)*theSBTables->fILDeltaElEnergy;
      elEnergyIndx     = (int)val;
      double pIndxH    = val-elEnergyIndx;
      // check if we are at limiting case: lower edge e- energy < gam-gut
      if (theSBTables->fElEnergyVect[elEnergyIndx] <= theGamCut) {
        // recompute the probability of taking the higher e- energy bin()
        pIndxH   = (theLogEkin-theLogGamCut)/(theSBTables->fLElEnergyVect[elEnergyIndx+1]-theLogGamCut);
        isCorner = true;
      }
      //
      if (rnge->flat()<pIndxH) {
        ++elEnergyIndx;      // take the table at the higher e- energy bin
      } else if (isCorner) { // take the table at the lower  e- energy bin
        // special sampling need to be done if lower edge e- energy < gam-gut:
        // actually, we "sample" from a table "built" at the gamm-cut i.e. delta
        isSimply = true;
      }
    }
    // compute the start index of the sampling table data for this `elEnergyIndx`
    const int   minEIndx = (int)(theSBTables->fSBTableData[iStart+1]);
    const int numGamCuts = (int)(theSBTables->fSBTableData[iStart+3]);
    const int   sizeOneE = (int)(numGamCuts + 3*theSBTables->fNumKappa);
    const int   iSTStart = iStart + 4 + (elEnergyIndx-minEIndx)*sizeOneE;
    // the minimum value of the cumulative (that corresponds to the kappa-cut value)
    const double    minV = theSBTables->fSBTableData[iSTStart+iGamCut];
    // the start of the table with the 54 kappa-cumulative and par-A and par-B values.
    const double* stData = &(theSBTables->fSBTableData[iSTStart+numGamCuts]);
    // some transfomrmtion variables used in the looop
    //  const double lCurKappaC  = theLogGamCut-theLogEkin;
    //  const double lUsedKappaC = theLogGamCut-theSBTables->fLElEnergyVect[elEnergyIndx];
    const double lKTrans = (theLogGamCut-theLogEkin)/(theLogGamCut-theSBTables->fLElEnergyVect[elEnergyIndx]);
    // dielectric (always) and e+ correction suppressions (if the primary is e+)
    const double primETot = thePrimEkin + kElectronMassC2;
    const double dielSupConst = theMData.fDensityCorFactor*primETot*primETot;
    double suppression = 1.0;
    double rndm[2];
    // rejection loop starts here (rejection only for the diel-supression)
    double eGamma = 0.0;
    do {
      rnge->flatArray(2, rndm);
      double kappa = 1.0;
      if (!isSimply) {
        const double cumRV  = rndm[0]*(1.0-minV)+minV;
        // find lower index of the values in the Cumulative Function: use linear
        // instead of binary search because it's faster in our case
        // note: every 3rd value of `stData` is the cumulative for the corresponding kappa grid values
        const int cumLIndx3 = LinSearch(stData, theSBTables->fNumKappa, cumRV) - 3;
        const int  cumLIndx = cumLIndx3/3;
        const double   cumL = stData[cumLIndx3];
        const double     pA = stData[cumLIndx3+1];
        const double     pB = stData[cumLIndx3+2];
        const double   cumH = stData[cumLIndx3+3];
        const double    lKL = theSBTables->fLKappaVect[cumLIndx];
        const double    lKH = theSBTables->fLKappaVect[cumLIndx+1];
        const double    dm1 = (cumRV-cumL)/(cumH-cumL);
        const double    dm2 = (1.0+pA+pB)*dm1;
        const double    dm3 = 1.0+dm1*(pA+pB*dm1);
        // kappa sampled at E_i e- energy
        const double lKappa = lKL+dm2/dm3*(lKH-lKL);
        // transform lKappa to [log(gcut/ekin),0] form [log(gcut/E_i),0]
        kappa  = G4HepEmExp(lKappa*lKTrans);
      } else {
        kappa = 1.0-rndm[0]*(1.0-theGamCut/thePrimEkin);
      }
      // compute the emitted photon energy: k
      eGamma = kappa*thePrimEkin;
      const double invEGamma = 1.0/eGamma;
      // compute dielectric suppression: 1/(1+[gk_p/k]^2)
      suppression = 1.0/(1.0+dielSupConst*invEGamma*invEGamma);
      // add positron correction if particle is e+
      if (!iselectron) {
        const double     e1 = thePrimEkin - theGamCut;
        const double iBeta1 = (e1 + kElectronMassC2) / std::sqrt(e1*(e1 + 2.0*kElectronMassC2));
        const double     e2 = thePrimEkin - eGamma;
        const double iBeta2 = (e2 + kElectronMassC2) / std::sqrt(e2*(e2 + 2.0*kElectronMassC2));
        const double    dum = kAlpha*k2Pi*dZet*(iBeta1 - iBeta2);
        suppression = (dum > -12.) ? suppression*G4HepEmExp(dum) : 0.;
      }
    } while (rndm[1] > suppression);
    return eGamma;
  }

  // Sampling of the energy transferred to the emitted photon using the Bethe-Heitler
  // DCS.
  template <typename RandomEngine>
  G4HepEmHostDevice
  static double SampleETransferRB(struct G4HepEmData* hepEmData, double thePrimEkin, double theLogEkin,
                                  int theMCIndx, RandomEngine* rnge, bool iselectron) {
    const G4HepEmMCCData& theMCData = hepEmData->fTheMatCutData->fMatCutData[theMCIndx];
    const double          theGamCut = theMCData.fSecGamProdCutE;
    //  const double       theLogGamCut = theMCData.fLogSecGamCutE;

    // get the material data
    const G4HepEmMatData&  theMData = hepEmData->fTheMaterialData->fMaterialData[theMCData.fHepEmMatIndex];
    // sample target element
    const G4HepEmElectronData*  theElData = iselectron ? hepEmData->fTheElectronData : hepEmData->fThePositronData;
    const int elemIndx = (theMData.fNumOfElement > 1)
                                               ? SelectTargetAtom(theElData, theMCIndx, thePrimEkin, theLogEkin, rnge->flat(), false)
                                               : 0;
    const int     iZet = theMData.fElementVect[elemIndx];
    const double  dZet = (double)iZet;
    const G4HepEmElemData& theElemData = hepEmData->fTheElementData->fElementData[G4HepEmMin(iZet, hepEmData->fTheElementData->fMaxZet)];
    //
    // == Sampling of the emitted photon energy
    // - compute lpm energy
    const double densityFactor = kMigdalConst * theMData.fElectronDensity;
    const double     lpmEnergy = kLPMconstant * theMData.fRadiationLength;
    // threshold for LPM effect (i.e. below which LPM hidden by density effect)
    const double  lpmEnergyLim = std::sqrt(densityFactor) * lpmEnergy;
    // compute the density, i.e. dielectric suppression correction factor
    const double thePrimTotalE = thePrimEkin + kElectronMassC2;
    const double   densityCorr = densityFactor * thePrimTotalE * thePrimTotalE;
    // LPM effect is turned off if thePrimTotalE < lpmEnergyLim
    const bool     isLPMActive = (thePrimTotalE > lpmEnergyLim) ;
    // compute/set auxiliary variables used in the energy transfer sampling
    const double      zFactor1 = theElemData.fZFactor1;
    const double      zFactor2 = (1.+1./dZet)/12.;
    const double    rejFuncMax = zFactor1 + zFactor2;
    // min and range of the transformed variable: x(k) = ln(k^2+k_p^2) that is in [ln(k_c^2+k_p^2), ln(E_k^2+k_p^2)]
    const double xmin   = G4HepEmLog( theGamCut*theGamCut     + densityCorr );
    const double xrange = G4HepEmLog( thePrimEkin*thePrimEkin + densityCorr ) - xmin;
    // sampling the emitted gamma energy
    double rndm[2];
    double eGamma, funcVal;
    do {
      rnge->flatArray(2, rndm);
      eGamma = std::sqrt( G4HepEmMax( G4HepEmExp( xmin + rndm[0] * xrange ) - densityCorr, 0.0 ) );
      // evaluate the DCS at this emitted gamma energy
      const double y     = eGamma / thePrimTotalE;
      const double onemy = 1.-y;
      const double dum0  = 0.25*y*y;
      if ( isLPMActive ) { // DCS: Bethe-Heitler in complete screening and LPM suppression
        // evaluate LPM functions (combined with the Ter-Mikaelian effect)
        double funcGS, funcPhiS, funcXiS;
        EvaluateLPMFunctions(funcXiS, funcGS, funcPhiS, eGamma, thePrimTotalE, lpmEnergy, theElemData.fZet23, theElemData.fILVarS1, theElemData.fILVarS1Cond, densityCorr, 1.0);
        const double term1 = funcXiS * ( dum0 * funcGS + (onemy+2.0*dum0) * funcPhiS );
        funcVal = term1*zFactor1 + onemy*zFactor2;
      } else {  // DCS: Bethe-Heitler without LPM suppression and complete screening only if Z<5 (becaue TF screening is not vaild for low Z)
        const double dum1 = onemy + 3.*dum0;
        if ( iZet < 5 ) { // DCS: complete screening
          funcVal = dum1 * zFactor1 + onemy * zFactor2;
        } else { // DCS: analytical approximations to the universal screening functions (based on TF model of atom)
          const double dum2 = y / ( thePrimTotalE - eGamma );
          const double gam  = dum2 * 100.*kElectronMassC2 / theElemData.fZet13;
          const double eps  = gam / theElemData.fZet13;
          // evaluate the screening functions (TF model of the atom, Tsai's aprx.):

          const double gam2 = gam*gam;
          const double phi1 = 16.863-2.0*G4HepEmLog(1.0+0.311877*gam2)+2.4*G4HepEmExp(-0.9*gam)+1.6*G4HepEmExp(-1.5*gam);
          const double phi2 = 2.0/(3.0+19.5*gam+18.0*gam2);    // phi1-phi2
          const double eps2 = eps*eps;
          const double psi1 = 24.34-2.0*G4HepEmLog(1.0+13.111641*eps2)+2.8*G4HepEmExp(-8.0*eps)+1.2*G4HepEmExp(-29.2*eps);
          const double psi2 = 2.0/(3.0+120.0*eps+1200.0*eps2); //psi1-psi2
          //
          const double logZ = theElemData.fLogZ;
          const double Fz   = logZ/3. + theElemData.fCoulomb;
          const double invZ = 1./dZet;
          funcVal = dum1*((0.25*phi1-Fz) + (0.25*psi1-2.*logZ/3.)*invZ) +  0.125*onemy*(phi2 + psi2*invZ);
        }
      }
      funcVal = G4HepEmMax( 0.0, funcVal);
    } while ( funcVal < rejFuncMax * rndm[1] );
    return eGamma;
  }

  // Target atom selector for the above bremsstrahlung intercations in case of
  // materials composed from multiple elements.
  G4HepEmHostDevice
  static int SelectTargetAtom(const struct G4HepEmElectronData* elData, const int imc, const double ekin,
                              const double lekin, const double urndn, const bool isbremSB);

  template <typename RandomEngine>
  G4HepEmHostDevice
  static void SampleDirections(const double thePrimEkin, const double theSecGammaEkin, double* theSecGammaDir,
                               double* thePrimElecDir, RandomEngine* rnge);


  // Simple linear search (with step of 3!) used in the photon energy sampling part
  // of the SB (Seltzer-Berger) brem model.
  G4HepEmHostDevice
  static int LinSearch(const double* vect, const int size, const double val);
};

#endif // G4HepEmElectronInteractionBrem_HH
