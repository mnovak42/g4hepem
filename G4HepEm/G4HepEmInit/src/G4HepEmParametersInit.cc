
#include "G4HepEmParametersInit.hh"

#include "G4HepEmParameters.hh"

// g4 include
#include "G4Version.hh"
#include "G4EmParameters.hh"
#include "G4RegionStore.hh"
#include "G4MscStepLimitType.hh"
#include "G4SystemOfUnits.hh"

// NOTE: here we set all parameters according to their Geant4 values then
//       later, in the `G4HepEmRunManager` from where this method is invoked from,
//       we apply possible per-region configurations on the top of this.
void InitHepEmParameters(struct G4HepEmParameters* hepEmPars) {
  // tracking cut for e- in internal Geant4 energy units
  hepEmPars->fElectronTrackingCut = G4EmParameters::Instance()->LowestElectronEnergy();
  hepEmPars->fGammaTrackingCut    = 0.0;

  // energy loss table (i.e. dE/dx) related paramaters
  hepEmPars->fMinLossTableEnergy   = G4EmParameters::Instance()->MinKinEnergy();
  hepEmPars->fMaxLossTableEnergy   = G4EmParameters::Instance()->MaxKinEnergy();
  hepEmPars->fNumLossTableBins     = G4EmParameters::Instance()->NumberOfBins();

  // e-/e+ related auxilary parameters:
  // energy limit between the 2 models (Seltzer-Berger and RelBrem) used for e-/e+
  hepEmPars->fElectronBremModelLim = 1.0*CLHEP::GeV;

  // flag to indicate if the e+ correction to the MSC theta0 angle should be used
  // note: em parameter for this available only in Geant4 version >= 11.1
#if G4VERSION_NUMBER >= 1110
  hepEmPars->fIsMSCPositronCor = G4EmParameters::Instance()->MscPositronCorrection();
#else
  hepEmPars->fIsMSCPositronCor = true;
#endif
  hepEmPars->fIsMSCDisplacement = G4EmParameters::Instance()->LateralDisplacement();

  // get the number of detector regions and allocate the per-region data array
  int numRegions = G4RegionStore::GetInstance()->size();
  hepEmPars->fNumRegions = numRegions;
  hepEmPars->fParametersPerRegion = new G4HepEmRegionParmeters[numRegions];

  // set default values for all regions (might be changed after this init)
  for (int i=0; i<G4RegionStore::GetInstance()->size(); ++i) {
    G4HepEmRegionParmeters& rDat = hepEmPars->fParametersPerRegion[i];
    // std::cout << " [" << i << "] Regin name = " << (*G4RegionStore::GetInstance())[i]->GetName() << std::endl;

    // NOTE: these values are hidden inside G4EmParameters::G4EmExtraParameters!!!
    rDat.fFinalRange    = 1.0*CLHEP::mm;
    rDat.fDRoverRange   = 0.2;
    rDat.fLinELossLimit = G4EmParameters::Instance()->LinearLossLimit();

    rDat.fMSCRangeFactor  = G4EmParameters::Instance()->MscRangeFactor();
    rDat.fMSCSafetyFactor = G4EmParameters::Instance()->MscSafetyFactor();

    rDat.fIsMSCMinimalStepLimit = (G4MscStepLimitType::fMinimal == G4EmParameters::Instance()->MscStepLimitType());

    rDat.fIsELossFluctuation = G4EmParameters::Instance()->LossFluctuation();

    rDat.fIsMultipleStepsInMSCTrans = true;

    rDat.fIsApplyCuts = G4EmParameters::Instance()->ApplyCuts();
  }
}
