#include "G4HepEmStateInit.hh"

#include "G4HepEmState.hh"
#include "G4HepEmParameters.hh"
#include "G4HepEmData.hh"

#include "G4HepEmParametersInit.hh"
#include "G4HepEmMaterialInit.hh"
#include "G4HepEmElectronInit.hh"
#include "G4HepEmGammaInit.hh"

void InitG4HepEmState(struct G4HepEmState* hepEmState)
{
  // Initialize parameters
  hepEmState->fParameters = new G4HepEmParameters;
  InitHepEmParameters(hepEmState->fParameters);

  // Initialize data and fill each subtable using its initialize function
  hepEmState->fData = new G4HepEmData;
  InitG4HepEmData(hepEmState->fData);

  InitMaterialAndCoupleData(hepEmState->fData, hepEmState->fParameters);

  // electrons, positrons
  InitElectronData(hepEmState->fData, hepEmState->fParameters, true);
  InitElectronData(hepEmState->fData, hepEmState->fParameters, false);

  InitGammaData(hepEmState->fData, hepEmState->fParameters);
}