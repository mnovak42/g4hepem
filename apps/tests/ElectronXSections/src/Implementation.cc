
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



bool TestXSectionData ( const struct G4HepEmData* hepEmData, bool iselectron ) {
  bool isPassed     = true;
  // number of mat-cut and kinetic energy pairs go generate and test
  int  numTestCases = 32768;
  // set up an rng to get mc-indices on [0,numMCData)
  std::random_device rd;
  std::mt19937 gen(rd());
  gen.seed(0); // fix seed
  std::uniform_real_distribution<> dis(0, 1.0);
  // get ptr to the G4HepEmElectronData and G4HepEmMatCutData structures
  const G4HepEmElectronData* theElectronData = hepEmData->fTheElectronData;
  const int numELossData = theElectronData->fELossEnergyGridSize;
  const G4HepEmMatCutData*   theMatCutData   = hepEmData->fTheMatCutData;
  const int numMCData    = theMatCutData->fNumMatCutData;  
  // allocate memory (host) to store the generated test cases:
  //  - the numTestCases, material-cut index and kinetic energy combinations
  // and the results:
  //  - the numTestCases, restricted macroscopic cross sction for ionisation, bremsstrahlung 
  //    evaluated at test cases.
  int*    tsInImc         = new int[numTestCases];
  double* tsInEkinIoni    = new double[numTestCases];
  double* tsInLogEkinIoni = new double[numTestCases];
  double* tsInEkinBrem    = new double[numTestCases];
  double* tsInLogEkinBrem = new double[numTestCases];
  double* tsOutResMXIoni  = new double[numTestCases];  
  double* tsOutResMXBrem  = new double[numTestCases];  
  // the maximum (+2%) primary particle kinetic energy that is covered by the simulation (100 TeV by default)
  const double    maxEKin = 1.02*theElectronData->fELossEnergyGrid[numELossData-1];
  for (int i=0; i<numTestCases; ++i) { 
    int imc            = (int)(dis(gen)*numMCData);
    tsInImc[i]         = imc;
    // == Ionisation: 
    // get the min/max of the possible prirmary e-/e+ kinetic energies at which
    // the restricted interacton can happen in this material-cuts (use +- 2% out of range)
    double secElCutE   = theMatCutData->fMatCutData[imc].fSecElProdCutE;
    double minEKin     = iselectron ? 0.98*2.0*secElCutE : 0.98*secElCutE;
    // generate a unifomly random kinetic energy point in the allowed (+- 2%) primary 
    // particle kinetic energy range on logarithmic scale
    double lMinEkin    = std::log(minEKin);
    double lEkinDelta  = std::log(maxEKin/minEKin);    
    tsInLogEkinIoni[i] = dis(gen)*lEkinDelta+lMinEkin;
    tsInEkinIoni[i]    = std::exp(tsInLogEkinIoni[i]);
    // == Bremsstrahlung: (the same with different limits)
    minEKin            = 0.98*theMatCutData->fMatCutData[imc].fSecGamProdCutE;
    lMinEkin           = std::log(minEKin);
    lEkinDelta         = std::log(maxEKin/minEKin);    
    tsInLogEkinBrem[i] = dis(gen)*lEkinDelta+lMinEkin;  
    tsInEkinBrem[i]    = std::exp(tsInLogEkinBrem[i]);      
  }
  //
  // Use a G4HepEmElectronManager object to evaluate the restricted macroscopic
  // cross sections for ionisation and bremsstrahlung for the test cases.
  G4HepEmElectronManager theElectronMgr;
  for (int i=0; i<numTestCases; ++i) { 
    tsOutResMXIoni[i] = theElectronMgr.GetRestMacXSec (theElectronData, tsInImc[i], tsInEkinIoni[i], tsInLogEkinIoni[i], true);
    tsOutResMXBrem[i] = theElectronMgr.GetRestMacXSec (theElectronData, tsInImc[i], tsInEkinBrem[i], tsInLogEkinBrem[i], false);
  }


#ifdef G4HepEm_CUDA_BUILD  
  //
  // Perform the test case evaluations on the device
  double* tsOutResOnDeviceMXIoni = new double[numTestCases]; 
  double* tsOutResOnDeviceMXBrem = new double[numTestCases]; 
  TestResMacXSecDataOnDevice (hepEmData, tsInImc, tsInEkinIoni, tsInLogEkinIoni, tsInEkinBrem, tsInLogEkinBrem, tsOutResOnDeviceMXIoni, tsOutResOnDeviceMXBrem, numTestCases);
  for (int i=0; i<numTestCases; ++i) { 
//    std::cout << tsInEkinIoni[i] << " "<<tsOutResMXIoni[i] << " " << tsOutResOnDeviceMXIoni[i] << " " <<tsInEkinBrem[i] << " " << tsOutResMXBrem[i] << " " << tsOutResOnDeviceMXBrem[i] << std::endl;
    if ( std::abs( 1.0 - tsOutResMXIoni[i]/tsOutResOnDeviceMXIoni[i] ) > 1.0E-14 ) {
      isPassed = false;
      std::cerr << "\n*** ERROR:\nRestricted Macroscopic Cross Section data: G4HepEm Host vs Device (Ioni) mismatch: " << tsOutResMXIoni[i] << " != " << tsOutResOnDeviceMXIoni[i] << " ( i = " << i << " imc  = " << tsInImc[i] << " ekin =  " << tsInEkinIoni[i] << ") " << std::endl; 
      break;
    }
    if ( std::abs( 1.0 - tsOutResMXBrem[i]/tsOutResOnDeviceMXBrem[i] ) > 1.0E-14 ) {
      isPassed = false;
      std::cerr << "\n*** ERROR:\nRestricted Macroscopic Cross Section data: G4HepEm Host vs Device (Brem) mismatch: " << tsOutResMXBrem[i] << " != " << tsOutResOnDeviceMXBrem[i] << " ( i = " << i << " imc  = " << tsInImc[i] << " ekin =  " << tsInEkinIoni[i] << ") " << std::endl; 
      break;
    }
  }
  // 
  delete [] tsOutResOnDeviceMXIoni;
  delete [] tsOutResOnDeviceMXBrem;
#endif // G4HepEm_CUDA_BUILD  

  //
  // delete allocatd memeory
  delete [] tsInImc;
  delete [] tsInEkinIoni;
  delete [] tsInLogEkinIoni;
  delete [] tsInEkinBrem;
  delete [] tsInLogEkinBrem;
  delete [] tsOutResMXIoni;
  delete [] tsOutResMXBrem;

  return isPassed;
}

