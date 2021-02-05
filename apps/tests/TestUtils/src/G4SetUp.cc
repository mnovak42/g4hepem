
#include "G4SetUp.hh"

// Geant4 includes
#include "G4SystemOfUnits.hh"

#include "G4NistManager.hh"
#include "G4Material.hh"
#include "G4String.hh"

#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4Region.hh"

#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"
#include "G4Electron.hh"
#include "G4Positron.hh"
#include "G4Gamma.hh"
#include "G4Proton.hh"

#include "G4DataVector.hh"
#include "G4ProductionCuts.hh"
#include "G4MaterialCutsCouple.hh"
#include "G4ProductionCutsTable.hh"
#include "G4EmParameters.hh"


// builds a fake Geant4 geometry just to be able to produce material-cuts couple
void FakeG4Setup ( G4double prodCutInLength, G4int verbose) {
  //
  // --- Geometry definition: create the word
  G4double wDimX      = 0.6*mm;
  G4double wDimY      = 0.6*mm;
  G4double wDimZ      = 0.6*mm;
  G4Material* wMat    = G4NistManager::Instance()->FindOrBuildMaterial("G4_Galactic");
  G4Box*           sW = new G4Box ("Box",wDimX, wDimY, wDimZ);
  G4LogicalVolume* lW = new G4LogicalVolume(sW,wMat,"Box",0,0,0);
  G4PVPlacement*   pW = new G4PVPlacement(0,G4ThreeVector(),"Box",lW,0,false,0);
  //
  // --- Build all NIST materials and set a logical volume for each
  const std::vector<G4String>& namesMat = G4NistManager::Instance()->GetNistMaterialNames();
  const G4int     numMat = namesMat.size();
  const G4double  halfX  =  0.5/numMat;  // half width of one material-box
  const G4double     x0  = -0.5+halfX;   // start x-position of the first material-box
  for (int im=0; im<numMat; ++im) {
    G4Material*       mat = G4NistManager::Instance()->FindOrBuildMaterial(namesMat[im]);
    G4Box*             ss = new G4Box ("Box", halfX, 0.5, 0.5);
    G4LogicalVolume*   ll = new G4LogicalVolume(ss, mat, "Box", 0, 0, 0);
    new G4PVPlacement(0, G4ThreeVector(x0+im*halfX , 0, 0), "Box", ll, pW, false, 0);
  }
  //
  // --- Create particles that has secondary production threshold
  G4Gamma::Gamma();
  G4Electron::Electron();
  G4Positron::Positron();
  G4Proton::Proton();
  G4ParticleTable* partTable = G4ParticleTable::GetParticleTable();
  partTable->SetReadiness();
  //
  // --- Create production - cuts object and set the secondary production threshold
  G4double prodCutValue = prodCutInLength;
  G4DataVector cuts;
  cuts.push_back(prodCutValue);
  G4ProductionCuts* pcut = new G4ProductionCuts();
  pcut->SetProductionCut(cuts[0], 0); // set cut for gamma
  pcut->SetProductionCut(cuts[0], 1); // set cut for e-
  pcut->SetProductionCut(cuts[0], 2); // set cut for e+
  pcut->SetProductionCut(cuts[0], 3); // set cut for p+
  //
  // --- Create the material-cuts couple objects: first the for the word, then
  //     create default region, add this word material-cuts couple  then all others.
  G4MaterialCutsCouple* couple0 = new G4MaterialCutsCouple(wMat, pcut);
  couple0->SetIndex(0);
  //
  G4Region* reg = new G4Region("DefaultRegionForTheWorld");
  reg->AddRootLogicalVolume(lW);
  reg->UsedInMassGeometry(true);
  reg->SetProductionCuts(pcut);
  reg->RegisterMaterialCouplePair(wMat, couple0);
  for (G4int im=0; im<numMat; ++im) {
    G4Material*              mat = G4NistManager::Instance()->GetMaterial(im);
    G4MaterialCutsCouple* couple = new G4MaterialCutsCouple(mat, pcut);
    couple->SetIndex(im+1);
    reg->RegisterMaterialCouplePair(mat, couple);
  }
  // --- Update the couple tables
  G4ProductionCutsTable* theCoupleTable = G4ProductionCutsTable::GetProductionCutsTable();
  theCoupleTable->UpdateCoupleTable(pW);
  //
  if ( verbose>0 ) {
    G4cout << " === FakeG4Setup() completed: \n"
           << "     - number of G4MaterialCutsCouple objects built = " << numMat          << "     \n"
           << "     - with secondary production threshold          = " << prodCutInLength << " [mm]\n"
           << G4endl;
  }
}


// builds a fake Geant4 geometry just to be able to produce material-cuts couple
const G4MaterialCutsCouple* FakeG4Setup ( G4double prodCutInLength, const G4String& nistMatName, G4int verbose) {
  //
  // --- Find the required NIST material for the target
  G4Material* wMat = G4NistManager::Instance()->FindOrBuildMaterial(nistMatName);
  if (!wMat) {
    std::cerr << " *** ERROR in FakeG4Setup: unknown G4-NIST material `"
              << nistMatName << "`!"
              << std::endl;
    exit(-1);
  }
  //
  // --- Geometry definition: create the word i.e. the target
  G4double wDimX      = 0.6*mm;
  G4double wDimY      = 0.6*mm;
  G4double wDimZ      = 0.6*mm;
  G4Box*           sW = new G4Box ("Box",wDimX, wDimY, wDimZ);
  G4LogicalVolume* lW = new G4LogicalVolume(sW,wMat,"Box",0,0,0);
  G4PVPlacement*   pW = new G4PVPlacement(0,G4ThreeVector(),"Box",lW,0,false,0);
  //
  // --- Create particles that has secondary production threshold
  G4Gamma::Gamma();
  G4Electron::Electron();
  G4Positron::Positron();
  G4Proton::Proton();
  G4ParticleTable* partTable = G4ParticleTable::GetParticleTable();
  partTable->SetReadiness();
  //
  // --- Create production - cuts object and set the secondary production threshold
  G4double prodCutValue = prodCutInLength;
  G4DataVector cuts;
  cuts.push_back(prodCutValue);
  G4ProductionCuts* pcut = new G4ProductionCuts();
  pcut->SetProductionCut(cuts[0], 0); // set cut for gamma
  pcut->SetProductionCut(cuts[0], 1); // set cut for e-
  pcut->SetProductionCut(cuts[0], 2); // set cut for e+
  pcut->SetProductionCut(cuts[0], 3); // set cut for p+
  //
  // --- Create the material-cuts couple objects
  G4MaterialCutsCouple* couple0 = new G4MaterialCutsCouple(wMat, pcut);
  couple0->SetIndex(0);
  //
  G4Region* reg = new G4Region("DefaultRegionForTheWorld");
  reg->AddRootLogicalVolume(lW);
  reg->UsedInMassGeometry(true);
  reg->SetProductionCuts(pcut);
  reg->RegisterMaterialCouplePair(wMat, couple0);
  // --- Update the couple tables
  G4ProductionCutsTable* theCoupleTable = G4ProductionCutsTable::GetProductionCutsTable();
  theCoupleTable->UpdateCoupleTable(pW);
  //
  if ( verbose>0 ) {
    G4cout << " === FakeG4Setup() completed: \n"
           << "     - number of G4MaterialCutsCouple objects built = " << 1              << "     \n"
           << "     - with secondary production threshold          = " << prodCutInLength << " [mm]\n"
           << G4endl;
  }
  // return with a pointer to teh registered material-cuts couple
  return couple0;
}
