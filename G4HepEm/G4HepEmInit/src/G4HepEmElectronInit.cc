
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
#include "G4UrbanMscModel.hh"

#include "G4VCrossSectionDataSet.hh"
#include "G4CrossSectionDataStore.hh"
#include "G4ElectroNuclearCrossSection.hh"

#include "G4HepEmMatCutData.hh"
#include "G4HepEmMaterialData.hh"
#include "G4HepEmElementData.hh"

#include <iostream>


void InitElectronData(struct G4HepEmData* hepEmData, struct G4HepEmParameters* hepEmPars,
                      bool iselectron, int verbose) {
  // clean previous G4HepEmElectronData (if any)
  //
  // create G4Models for e- or for e+
  G4ParticleDefinition* g4PartDef = G4Positron::Positron();
  if (iselectron) {
    g4PartDef = G4Electron::Electron();
  }
  if (verbose > 1) std::cout << "     ---  InitElectronData ... " << std::endl;
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
  // get cuts for secondary gamma
  const G4DataVector* theGamCuts = static_cast<const G4DataVector*>(theCoupleTable->GetEnergyCutsVector(0));
  // ACTIVATE sampling tables
  modelSB->Initialise(g4PartDef, *theGamCuts);
  //
  // 3. High energy brem. model with LPM correction (LPM flag is set by G4EmParameters):
  // --- used on [1GeV : E_max]
  G4eBremsstrahlungRelModel* modelRB = new G4eBremsstrahlungRelModel();
  modelRB->SetLowEnergyLimit(energyLimit);
  modelRB->SetHighEnergyLimit(emModelEMax);
  modelRB->SetSecondaryThreshold(G4EmParameters::Instance()->BremsstrahlungTh());
  modelRB->Initialise(g4PartDef, *theGamCuts);
  //
  // 4. Urban msc model:
  // --- used on [E_min : E_max]
  G4UrbanMscModel* modelUMSC = new G4UrbanMscModel();
  modelUMSC->SetLowEnergyLimit(emModelEMin);
  modelUMSC->SetHighEnergyLimit(emModelEMax);
  modelUMSC->Initialise(g4PartDef, *theGamCuts); // second argument is not used

  //
  // 5. Electron - and positorn nuclear cross section
  // --- used on [100 MeV : E_max] (same for e-/e+)
  G4VCrossSectionDataSet* xs = new G4ElectroNuclearCrossSection;
  xs->BuildPhysicsTable(*g4PartDef);
  G4CrossSectionDataStore hadENucXSDataStore;
  hadENucXSDataStore.AddDataSet(xs);

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
  if (verbose > 1) std::cout << "     ---  BuildELossTables ..." << std::endl;
  BuildELossTables(modelMB, modelSB, modelRB, hepEmData, hepEmPars, iselectron);
  // build macroscopic cross section data (mat-cut dependent ioni and brem)
  if (verbose > 1) std::cout << "     ---  BuildLambdaTables ... " << std::endl;
  BuildLambdaTables(modelMB, modelSB, modelRB, hepEmData, hepEmPars, iselectron);
  // build macroscopic cross section data (mat dependent electron -, positron - nuclear)
  BuildNuclearLambdaTables(&hadENucXSDataStore, hepEmData, hepEmPars, iselectron);
  // build macroscopic first transport cross section data (used by Urban msc)
  if (verbose > 1) std::cout << "     ---  BuildTransportXSectionTables ... " << std::endl;
  BuildTransportXSectionTables(modelUMSC, hepEmData, hepEmPars, iselectron);
  // build element selectors
  if (verbose > 1) std::cout << "     ---  BuildElementSelectorTables ... " << std::endl;
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
    if (verbose > 1) std::cout << "     ---  BuildSBBremTables ... " << std::endl;
    BuildSBBremSTables(hepEmData, hepEmPars, modelSB);
  }

  // delete all g4 models
  delete modelMB;
  delete modelSB;
  delete modelRB;
  delete modelUMSC;
}
