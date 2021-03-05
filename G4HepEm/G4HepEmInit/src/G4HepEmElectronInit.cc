
#include "G4HepEmElectronInit.hh"

#include "G4HepEmData.hh"
#include "G4HepEmParameters.hh"
#include "G4HepEmElectronData.hh"

#include "G4HepEmElectronTableBuilder.hh"

// g4 includes
#include "G4EmParameters.hh"
#include "G4ProductionCutsTable.hh"

#include "G4ParticleDefinition.hh"
#include "G4Electron.hh"
#include "G4Positron.hh"
#include "G4SystemOfUnits.hh"

#include "G4DataVector.hh"
#include "G4MollerBhabhaModel.hh"
#include "G4SeltzerBergerModel.hh"
#include "G4eBremsstrahlungRelModel.hh"

#include "G4HepEmMatCutData.hh"
#include "G4HepEmMaterialData.hh"
#include "G4HepEmElementData.hh"

#include <iostream>


void InitElectronData(struct G4HepEmData* hepEmData, struct G4HepEmParameters* hepEmPars,
                      bool iselectron) {
  // clean previous G4HepEmElectronData (if any)
  //
  // create G4Models for e- or for e+
  G4ParticleDefinition* g4PartDef = G4Positron::Positron();
  if (iselectron) {
    g4PartDef = G4Electron::Electron();
  }
  std::cout << "     ---  InitElectronData ... " << std::endl;
  // Min/Max energies of the EM model (same as for the loss-tables)
  G4double emModelEMin = G4EmParameters::Instance()->MinKinEnergy();
  G4double emModelEMax = G4EmParameters::Instance()->MaxKinEnergy();
  // we will need the couple table to get the cuts
  G4ProductionCutsTable* theCoupleTable = G4ProductionCutsTable::GetProductionCutsTable();
  //
  // 1. Moller-Bhabha model for ionisation:
  // --- used on [E_min : E_max]
  G4MollerBhabhaModel*  modelMB = new G4MollerBhabhaModel();
  modelMB->SetLowEnergyLimit(emModelEMin);
  modelMB->SetHighEnergyLimit(emModelEMax);
  // get cuts for secondary e-
  const G4DataVector* theElCuts = static_cast<const G4DataVector*>(theCoupleTable->GetEnergyCutsVector(1));
  modelMB->Initialise(g4PartDef, *theElCuts);
  //
  //
  // 2. Seltzer-Berger numerical DCS based model for brem. photon emission:
  // --- used on [E_min : 1 GeV] note: data are available from 1 keV to 1 GeV
  G4SeltzerBergerModel* modelSB = new G4SeltzerBergerModel();
  modelSB->SetLowEnergyLimit(emModelEMin);
  G4double energyLimit = std::min(modelSB->HighEnergyLimit(), hepEmPars->fElectronBremModelLim);
  modelSB->SetHighEnergyLimit(energyLimit);
  modelSB->SetSecondaryThreshold(G4EmParameters::Instance()->BremsstrahlungTh());
  modelSB->SetLPMFlag(false);
  // get cuts for secondary gamma
  const G4DataVector* theGamCuts = static_cast<const G4DataVector*>(theCoupleTable->GetEnergyCutsVector(0));
  // ACTIVATE sampling tables
  modelSB->Initialise(g4PartDef, *theGamCuts);
  //
  // 3. High energy brem. model with LPM correction:
  // --- used on [1GeV : E_max]
  G4eBremsstrahlungRelModel* modelRB = new G4eBremsstrahlungRelModel();
  modelRB->SetLowEnergyLimit(energyLimit);
  modelRB->SetHighEnergyLimit(emModelEMax);
  modelRB->SetSecondaryThreshold(G4EmParameters::Instance()->BremsstrahlungTh());
  modelRB->SetLPMFlag(true);
  modelRB->Initialise(g4PartDef, *theGamCuts);
  //
  // === Use the G4HepEmElectronTableBuilder to build all data tables used at
  //     run time: e-loss, macroscopic cross section tables and target element
  //     selectors for each models.
  //
  // allocate (the ELossData part of) the ElectronData (NOTE: shallow only,
  // BuildELossTables will complete the allocation) but cleans the memory of the
  // hepEmData->fTheElectronData
  if (iselectron) {
    AllocateElectronData(&(hepEmData->fTheElectronData));
  } else {
    AllocateElectronData(&(hepEmData->fThePositronData));
  }
  // build energy loss data
  std::cout << "     ---  BuildELossTables ..." << std::endl;
  BuildELossTables(modelMB, modelSB, modelRB, hepEmData, hepEmPars, iselectron);
  // build macroscopic cross section data
  std::cout << "     ---  BuildLambdaTables ... " << std::endl;
  BuildLambdaTables(modelMB, modelSB, modelRB, hepEmData, hepEmPars, iselectron);
  // build element selectors
  std::cout << "     ---  BuildElementSelectorTables ... " << std::endl;
  BuildElementSelectorTables(modelMB, modelSB, modelRB, hepEmData, hepEmPars, iselectron);
  //
  // === Initialize the interaction description part of all models
  //
  // init sampling for MB, SB and RB models:
  // - nothing to init for MB
  // - SB: create a SB-table builder, build the S-tables and read them to produce
  //       the G4HepEmSBTables data structure (SB-table are the same fo -e and e+
  //       so we should build them only once)
  if (!hepEmData->fTheSBTableData) {
    std::cout << "     ---  BuildSBBremTables ... " << std::endl;
    BuildSBBremSTables(hepEmData, hepEmPars, modelSB);
  }

  // delete all g4 models
  delete modelMB;
  delete modelSB;
  delete modelRB;
}
