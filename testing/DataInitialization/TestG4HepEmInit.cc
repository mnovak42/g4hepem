// local (and TestUtils) includes
#include "TestUtils/G4SetUp.hh"
#include "TestUtils/G4HepEmDataComparison.hh"

// G4 includes
#include "globals.hh"
#include "G4SystemOfUnits.hh"
#include "Randomize.hh"

// G4HepEm includes
#include "G4HepEmRunManager.hh"
#include "G4HepEmRandomEngine.hh"
#include "G4HepEmStateInit.hh"
#include "G4HepEmState.hh"
#include "G4HepEmParameters.hh"
#include "G4HepEmData.hh"

int main() {
  // --- Set up a fake G4 geometry with including all pre-defined NIST materials
  //     to produce the G4MaterialCutsCouple objects.
  //
  // secondary production threshold in length
  const G4double secProdThreshold = 0.7*mm;
  FakeG4Setup (secProdThreshold, true);

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
  //     method for e- (could be any of e-: 0; e+: 1; or gamma: 2).
  G4HepEmRunManager* runMgr = new G4HepEmRunManager ( true );
  runMgr->Initialize ( new G4HepEmRandomEngine(G4Random::getTheEngine()), 0 );
  runMgr->Initialize ( new G4HepEmRandomEngine(G4Random::getTheEngine()), 1 );
  runMgr->Initialize ( new G4HepEmRandomEngine(G4Random::getTheEngine()), 2 );

  G4HepEmParameters* rmParams = runMgr->GetHepEmParameters();
  G4HepEmData* rmData = runMgr->GetHepEmData();

  if(rmParams == nullptr)
  {
    std::cerr << "Failed to create G4HepEmParameters from G4HepEmRunManager" << std::endl;
    return 1;
  }
  if(rmData == nullptr)
  {
    std::cerr << "Failed to create G4HepEmData from G4HepEmRunManager" << std::endl;
    return 1;
  }

  // Now via G4HepEmInit
  G4HepEmState initState;
  InitG4HepEmState(&initState);
  if(initState.fParameters == nullptr)
  {
    std::cerr << "Failed to create G4HepEmParameters from G4HepEmInit" << std::endl;
    return 1;
  }
  if(initState.fData == nullptr)
  {
    std::cerr << "Failed to create G4HepEmData from G4HepEmInit" << std::endl;
    return 1;
  }

  // Comparison
  if(*rmParams != *(initState.fParameters))
  {
    std::cerr << "G4HepEmParameters from G4HepEmRunManager and G4HepEmInit are not numerically equal" << std::endl;
    return 1;
  }

  if(*rmData != *(initState.fData))
  {
    std::cerr << "G4HepEmData from G4HepEmRunManager and G4HepEmInit are not numerically equal" << std::endl;
    return 1;
  }

  // Cleanup
  delete runMgr;
  delete initState.fParameters;
  delete initState.fData;

  return 0;
}
