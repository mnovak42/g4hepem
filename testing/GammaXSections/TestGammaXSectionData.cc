
#include "Declaration.hh"

// local (and TestUtils) includes
#include "TestUtils/G4SetUp.hh"

// G4 includes
#include "globals.hh"
#include "G4SystemOfUnits.hh"
#include "Randomize.hh"

// G4HepEm includes
#include "G4HepEmRunManager.hh"
#include "G4HepEmData.hh"
#include "G4HepEmCLHEPRandomEngine.hh"

int main() {
  int verbose = 1;
  //
  // --- Set up a fake G4 geometry with including all pre-defined NIST materials
  //     to produce the G4MaterialCutsCouple objects.
  //
  // secondary production threshold in length
  const G4double secProdThreshold = 0.7*mm;
  FakeG4Setup (secProdThreshold, verbose);
  // print out all material-cuts (needs the G4ProductionCutsTable.hh include)
//  G4ProductionCutsTable* theCoupleTable = G4ProductionCutsTable::GetProductionCutsTable();
//  theCoupleTable->DumpCouples();

  //
  // --- Initialise the `global` data structures of G4HepEm:
  //     - the `global`-s are the G4HepEmParameters, G4HepEmMatCutData, G4HepEmMaterialData,
  //       and G4HepEmElementData members of the G4HepEmData, top level data structure member
  //       of the `master` G4HepEmRunManager
  //     - these above data structures are constructed and initialised in the G4HepEmRunManager::InitializeGlobal()
  //       method, that is invoked when the `master` G4HepEmRunManager::Initialize() method is invoked
  //       first time (i.e. for the first particle)
  //     - this should be done after the Geant4 geometry is already initialized, since data will be
  //       extracted from the already initialized Geant4 obejcts such as G4ProductionCutsTable
  //
  //     Therefore, here we create a `master` G4HepEmRunManager and call its Initialize()
  //     method for gamma (e-: 0; e+: 1; or gamma: 2) that will init all global, plus the gamma
  //     related data.
  int g4HepEmParticleIndx = 2;
  G4HepEmRunManager* runMgr = new G4HepEmRunManager ( true );
  runMgr->Initialize ( new G4HepEmCLHEPRandomEngine(G4Random::getTheEngine()), g4HepEmParticleIndx );

  //
  // --- Make all G4HepEmData member available on the device (only if G4HepEm_CUDA_BUILD)
  //
#ifdef G4HepEm_CUDA_BUILD
  CopyG4HepEmDataToGPU ( runMgr->GetHepEmData() );
#endif // G4HepEm_CUDA_BUILD


  //
  // --- Invoke the test for gamma related Macroscopic Cross Section structure test(s):
  if ( !TestGammaXSectionData ( runMgr->GetHepEmData() ) ) {
    return 1;
  } else if ( verbose > 0 ) {
#ifdef G4HepEm_CUDA_BUILD
    std::cout << " === Gamma Macroscopic Cross Section Test: PASSING (HepEm HOST v.s. DEVICE) \n" << std::endl;
#else   // G4HepEm_CUDA_BUILD
    std::cout << " === Gamma Macroscopic Cross Section Test: PASSING (HepEm HOST) \n" << std::endl;
#endif  // G4HepEm_CUDA_BUILD
  }

  return 0;
}
