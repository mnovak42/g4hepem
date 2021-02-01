
#include "Declaration.hh"

// Geant4 includes
#include "G4SystemOfUnits.hh"

#include "G4NistManager.hh"
#include "G4Material.hh"
#include "G4String.hh"

#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4Region.hh"

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



// G4HepEm includes
#include "G4HepEmData.hh"
#include "G4HepEmMatCutData.hh"
#include "G4HepEmElectronData.hh"

// don't worry it's just for testing
#define private public
#include "G4HepEmElectronManager.hh"

#include <vector>
#include <cmath>
#include <random>


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



bool TestElossData ( const struct G4HepEmData* hepEmData, bool iselectron ) {
  bool isPassed     = true;
  // number of mat-cut and kinetic energy pairs go generate and test
  int  numTestCases = 32768;
  // number of mat-cut data i.e. G4HepEm mat-cut indices are in [0,numMCData)
  int  numMCData    = hepEmData->fTheMatCutData->fNumMatCutData;
  // set up an rng to get mc-indices on [0,numMCData)
  std::random_device rd;
  std::mt19937 gen(rd());
  gen.seed(0); // fix seed
  std::uniform_real_distribution<> dis(0, 1.0);
  // get ptr to the G4HepEmElectronData structure
  const G4HepEmElectronData* theElectronData = iselectron ? hepEmData->fTheElectronData : hepEmData->fThePositronData;
  // for the generation of test particle kinetic energy values:
  // - get the min/max values of the energy loss (related data) kinetic energy grid
  // - also the number of discrete kinetic energy grid points (used later)
  // - test particle kinetic energies will be generated uniformly random, on log
  //   kinetic energy scale, between +- 5 percent of the limits (in order to test
  //   below above grid limits cases as well)
  const int     numELossData = theElectronData->fELossEnergyGridSize;
  const double  minELoss     = 0.95*theElectronData->fELossEnergyGrid[0];
  const double  maxELoss     = 1.05*theElectronData->fELossEnergyGrid[numELossData-1];
  // allocate memory (host) to store the generated test cases:
  //  - the numTestCases, material-cut index and kinetic energy combinations
  // and the results:
  //  - the numTestCases, restricted dEdx, range and inverse-range values for the
  //    test cases.
  int*    tsInImc           = new int[numTestCases];
  double* tsInEkin          = new double[numTestCases];
  double* tsInLogEkin       = new double[numTestCases];
  double* tsOutResRange     = new double[numTestCases];
  double* tsOutResDEDX      = new double[numTestCases];
  double* tsOutResInvRange  = new double[numTestCases];
  // generate the test cases: mat-cut indices and kinetic energy combinations
  const double lMinELoss   = std::log(minELoss);
  const double lELossDelta = std::log(maxELoss/minELoss);
  for (int i=0; i<numTestCases; ++i) {
    tsInImc[i]     = (int)(dis(gen)*numMCData);
    tsInLogEkin[i] = dis(gen)*lELossDelta+lMinELoss;
    tsInEkin[i]    = std::exp(tsInLogEkin[i]);
  }
  //
  // Use a G4HepEmElectronManager object to evaluate the range, dedx and inverse-range
  // values for the test cases.
  G4HepEmElectronManager theElectronMgr;
  for (int i=0; i<numTestCases; ++i) {
    tsOutResRange[i]    = theElectronMgr.GetRestRange(theElectronData, tsInImc[i], tsInEkin[i], tsInLogEkin[i]);
    tsOutResDEDX[i]     = theElectronMgr.GetRestDEDX (theElectronData, tsInImc[i], tsInEkin[i], tsInLogEkin[i]);
    tsOutResInvRange[i] = theElectronMgr.GetInvRange (theElectronData, tsInImc[i], tsOutResRange[i]);
  }


#ifdef G4HepEm_CUDA_BUILD
  //
  // Perform the test case evaluations on the device
  double* tsOutResOnDeviceRange    = new double[numTestCases];
  double* tsOutResOnDeviceDEDX     = new double[numTestCases];
  double* tsOutResOnDeviceInvRange = new double[numTestCases];
  TestElossDataOnDevice (hepEmData, tsInImc, tsInEkin, tsInLogEkin, tsOutResOnDeviceRange, tsOutResOnDeviceDEDX, tsOutResOnDeviceInvRange, numTestCases, iselectron);
  for (int i=0; i<numTestCases; ++i) {
    if ( std::abs( 1.0 - tsOutResRange[i]/tsOutResOnDeviceRange[i] ) > 1.0E-14 ) {
      isPassed = false;
      std::cerr << "\n*** ERROR:\nEnergyLoss data: G4HepEm Host vs Device RANGE mismatch: " << tsOutResRange[i] << " != " << tsOutResOnDeviceRange[i] << " ( i = " << i << " imc  = " << tsInImc[i] << " ekin =  " << tsInEkin[i] << ") " << std::endl;
      break;
    }
    if ( std::abs( 1.0 - tsOutResDEDX[i]/tsOutResOnDeviceDEDX[i] ) > 1.0E-14 ) {
      isPassed = false;
      std::cerr << "\n*** ERROR:\nEnergyLoss data: G4HepEm Host vs Device dE/dx mismatch: "  << tsOutResDEDX[i] << " != " << tsOutResOnDeviceDEDX[i] << " ( i = " << i << " imc  = " << tsInImc[i] << " ekin =  " << tsInEkin[i] << ") " << std::endl;
      break;
    }
    if ( std::abs( 1.0 - tsOutResInvRange[i]/tsOutResOnDeviceInvRange[i] ) > 1.0E-14 ) {
      isPassed = false;
      std::cerr << "\n*** ERROR:\nEnergyLoss data: G4HepEm Host vs Device Inverse-RANGE mismatch: "  << tsOutResInvRange[i] << " != " << tsOutResOnDeviceInvRange[i] << " ( i = " << i << " imc  = " << tsInImc[i] << " ekin =  " << tsInEkin[i] << " range =  " << tsOutResRange[i]<< ") " << std::endl;
      break;
    }
  }
  //
  delete [] tsOutResOnDeviceRange;
  delete [] tsOutResOnDeviceDEDX;
  delete [] tsOutResOnDeviceInvRange;
#endif // G4HepEm_CUDA_BUILD


  //
  // delete allocatd memeory
  delete [] tsInImc;
  delete [] tsInEkin;
  delete [] tsInLogEkin;
  delete [] tsOutResRange;
  delete [] tsOutResDEDX;
  delete [] tsOutResInvRange;

  return isPassed;
}
