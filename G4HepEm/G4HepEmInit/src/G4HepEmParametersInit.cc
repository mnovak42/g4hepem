
#include "G4HepEmParametersInit.hh"

#include "G4HepEmParameters.hh"

// g4 include
#include "G4EmParameters.hh"
#include "G4SystemOfUnits.hh"

void InitHepEmParameters(struct G4HepEmParameters* hepEmPars) {
  // e-/e+ tracking cut in kinetic energy
  hepEmPars->fElectronTrackingCut = G4EmParameters::Instance()->LowestElectronEnergy();
  
  // energy loss table (i.e. dE/dx) related paramaters
  hepEmPars->fMinLossTableEnergy   = G4EmParameters::Instance()->MinKinEnergy();
  hepEmPars->fMaxLossTableEnergy   = G4EmParameters::Instance()->MaxKinEnergy();
  hepEmPars->fNumLossTableBins     = G4EmParameters::Instance()->NumberOfBins();
  
  hepEmPars->fFinalRange           = 1.0*CLHEP::mm;
  hepEmPars->fDRoverRange          = 0.2;
  hepEmPars->fLinELossLimit        = G4EmParameters::Instance()->LinearLossLimit();
  
  // e-/e+ related auxilary parameters:
  // energy limit between the 2 models (Seltzer-Berger and RelBrem) used for e-/e+
  hepEmPars->fElectronBremModelLim = 1.0*CLHEP::GeV; 
}
