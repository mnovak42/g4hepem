
#include "G4HepEmElectronInteractionBremSB.hh"

#include "G4HepEmTLData.hh"
#include "G4HepEmData.hh"
#include "G4HepEmMatCutData.hh"
#include "G4HepEmMaterialData.hh"
#include "G4HepEmElementData.hh"
#include "G4HepEmElectronData.hh"
#include "G4HepEmSBTableData.hh"

#include "G4HepEmElectronTrack.hh"
#include "G4HepEmGammaTrack.hh"
#include "G4HepEmConstants.hh"
#include "G4HepEmRunUtils.hh"


#include <cmath>
//#include <iostream>


// Bremsstrahlung interaction based on the numerical Seltzer-Berger DCS for the 
// emitted photon energy. 
// Used between 1 keV - 1 GeV primary e-/e+ kinetic energies.
// NOTE: the core part i.e. sampling the emitted photon energy is different than 
//       that in the G4SeltzerBergerModel. I implemented here my rejection free,
//       memory effcicient (tables only per Z and not per mat-cuts) sampling. 
//       Rejection is used only to account dielectric supression and e+ correction. 
void PerformElectronBremSB(G4HepEmTLData* tlData, struct G4HepEmData* hepEmData, bool iselectron) { 

  G4HepEmElectronTrack* thePrimaryElTrack = tlData->GetPrimaryElectronTrack(); 
  G4HepEmTrack* thePrimaryTrack = thePrimaryElTrack->GetTrack(); 
  
  double              thePrimEkin = thePrimaryTrack->GetEKin();
  const double        theLogEkin  = thePrimaryTrack->GetLogEKin();
  const int             theMCIndx = thePrimaryTrack->GetMCIndex();
  const G4HepEmMCCData& theMCData = hepEmData->fTheMatCutData->fMatCutData[theMCIndx];
  const double          theGamCut = theMCData.fSecGamProdCutE;
  const double       theLogGamCut = theMCData.fLogSecGamCutE;
  // return if intercation is not possible (should not happen)
  if (thePrimEkin <= theGamCut) return;
  // get the material data
  const G4HepEmMatData& theMData  = hepEmData->fTheMaterialData->fMaterialData[theMCData.fHepEmMatIndex];  
  // sample target element 
  const int elemIndx = (theMData.fNumOfElement > 1) 
                       ? SelectTargetAtomBrem(hepEmData->fTheElectronData, theMCIndx, thePrimEkin, 
                                              theLogEkin, tlData->GetRNGEngine()->flat(), true)
                       : 0;
  const int     iZet = theMData.fElementVect[elemIndx];
  const double  dZet = (double)iZet;
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
    if (tlData->GetRNGEngine()->flat()<pIndxH) {
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
  const double primPTot = std::sqrt(thePrimEkin * (primETot + kElectronMassC2));
  const double dielSupConst = theMData.fDensityCorFactor*primETot*primETot;
  double suppression = 1.0;
  double rndm[3];
  // rejection loop starts here (rejection only for the diel-supression)
  double eGamma = 0.0;
  do {
    tlData->GetRNGEngine()->flatArray(2, rndm);
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
      kappa  = std::exp(lKappa*lKTrans);
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
       suppression = (dum > -12.) ? suppression*std::exp(dum) : 0.;
     }
   } while (rndm[1] > suppression);
   // end of rejection loop: the photon energy is `eGamma`
   //
   // sample photon direction (modified Tsai sampling): 
   const double uMax = 2.0*(1.0 + thePrimEkin/kElectronMassC2);   
   double u;
   do {
     tlData->GetRNGEngine()->flatArray(3, rndm);
     const double uu = -std::log(rndm[0]*rndm[1]);
     u = (0.25 > rndm[2]) ? uu*1.6 : uu*0.533333333;  
   } while(u > uMax);
   const double cost = 1.0 - 2.0*u*u/(uMax*uMax);
   const double sint = std::sqrt((1.0-cost)*(1.0+cost));
   const double  phi = k2Pi*tlData->GetRNGEngine()->flat();
   // create secondary photon
   G4HepEmGammaTrack* secGamTrack = tlData->AddSecondaryGammaTrack();
   G4HepEmTrack*         secTrack = secGamTrack->GetTrack();
   secTrack->SetDirection(sint * std::cos(phi), sint * std::sin(phi), cost);
   double* theSecondaryDirection = secTrack->GetDirection();
   double* thePrimaryDirection   = thePrimaryTrack->GetDirection();
   // rotate back to refernce frame (G4HepEmRunUtils function)
   RotateToReferenceFrame(theSecondaryDirection, thePrimaryDirection);
   secTrack->SetEKin(eGamma);
   secTrack->SetParentID(thePrimaryTrack->GetID()); 

   //
   //
   // compute post-interaction kinematics of the primary e-/e+
   double elDirX = primPTot * thePrimaryDirection[0] - eGamma * theSecondaryDirection[0];
   double elDirY = primPTot * thePrimaryDirection[1] - eGamma * theSecondaryDirection[1];
   double elDirZ = primPTot * thePrimaryDirection[2] - eGamma * theSecondaryDirection[2];
   // normalisation
   const double norm = 1.0 / std::sqrt(elDirX * elDirX + elDirY * elDirY + elDirZ * elDirZ);
   // update primary track direction
   thePrimaryTrack->SetDirection(elDirX * norm, elDirY * norm, elDirZ * norm);
   // update primary track kinetic energy
   thePrimaryTrack->SetEKin(thePrimEkin - eGamma);
   // NOTE: the following usually set to very high energy so I don't include this.
   // if secondary gamma energy is higher than threshold(very high by default)
   // then stop tracking the primary particle and create new secondary e-/e+
   // instead of the primary
}


// Bremsstrahlung interaction based on the Bethe-Heitler DCS with several, but 
// most importantly, with LPM correction. 
// Used between 1 GeV - 100 TeV primary e-/e+ kinetic energies.
//void PerformElectronBremRB(G4HepEmTLData* tlData, struct G4HepEmData* hepEmData, bool iselectron) { 
//}


// should be called only for mat-cuts with more than one elements in their material
int SelectTargetAtomBrem(struct G4HepEmElectronData* elData, int imc, double ekin, double lekin, double urndn, bool isbremSB) {
  // start index for this mat-cut and this model (-1 is no elememnt selector i.e. single element material) 
  const int   indxStart = isbremSB 
                          ? elData->fElemSelectorBremSBStartIndexPerMatCut[imc] 
                          : elData->fElemSelectorBremRBStartIndexPerMatCut[imc];
  const double* theData = isbremSB 
                          ? &(elData->fElemSelectorBremSBData[indxStart])
                          : &(elData->fElemSelectorBremRBData[indxStart]);
  const int     numData = theData[0];
  const int     numElem = theData[1];
  const double    logE0 = theData[2];
  const double    invLD = theData[3];
  const double*   xdata = &(theData[4]);
  // make sure that $x \in  [x[0],x[ndata-1]]$
  const double   xv = std::max(xdata[0], std::min(xdata[numElem*(numData-1)], ekin));
  // compute the lowerindex of the x bin (idx \in [0,N-2] will be guaranted)
  const int idxEkin = std::max(0.0, std::min((lekin-logE0)*invLD, numData-2.0)); 
  // the real index position is idxEkin x numElem
  int   indx0 = idxEkin*numElem;
  int   indx1 = indx0+numElem;
  // linear interpolation
  const double   x1 = xdata[indx0++];
  const double   x2 = xdata[indx1++];
  const double   dl = x2-x1;
  const double    b = std::max(0., std::min(1., (xv - x1)/dl));
  int theElemIndex = 0;
  while (theElemIndex<numElem-1 && urndn > xdata[indx0+theElemIndex]+b*(xdata[indx1+theElemIndex]-xdata[indx0+theElemIndex])) { ++theElemIndex; }
  return theElemIndex;      
}


// find lower bin index of value: used in acse of CDF values i.e. val in [0,1)
// while vector elements in [0,1]
// note: every 3rd value of the vect contains the kappa-cumulutaive values
int LinSearch(const double* vect, const int size, const double val) {
  int i = 0;
  const int size3 = 3*size;
  while (i + 9 < size3) {
    if (vect [i + 0] > val) return i + 0;
    if (vect [i + 3] > val) return i + 3;
    if (vect [i + 6] > val) return i + 6;
    if (vect [i + 9] > val) return i + 9;
    i += 12;
  }
  while (i < size3) {
    if (vect [i] > val)
      break;
    i += 3;
  }
  return i;
}

