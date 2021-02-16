
#include "Declaration.hh"

// local (and TestUtils) includes
#include "TestBremArgs.hh"
#include "TestUtils/G4SetUp.hh"

// G4 includes
#include "globals.hh"
#include "Randomize.hh"
#include "G4MaterialCutsCouple.hh"

// G4HepEm includes
#include "G4HepEmRunManager.hh"
#include "G4HepEmData.hh"
#include "G4HepEmCLHEPRandomEngine.hh"


int main(int argc, char *argv[]) {
  int verbose = 1;
  //
  // --- Get input arguments
  struct BremArgs theArgs;
  GetBremArgs(argc, argv, theArgs);
  //
  const bool        theIsElectron         = (theArgs.fParticleName == "e-");
  const std::string theTargetMaterialName = theArgs.fMaterialName;
  const std::string theBremModelName      = theArgs.fBremModelName;
  const int         theTestType           = theArgs.fTestType;
  const int         theNumHistBins        = theArgs.fNumHistBins;
  const double      theNumSamples         = theArgs.fNumSamples;
  const double      thePrimaryKinEnergy   = theArgs.fPrimaryEnergy;
  const double      theSecondaryProdCut   = theArgs.fProdCutValue;

  //
  // --- Set up a fake G4 geometry with including (From TestUtils/G4SetUp):
  //       - test type = 0 or 1 :  the single pre-defined NIST material specified as target
  //       - test type = 2      :  all pre-defined NIST materials
  //     to produce the corresponding G4MaterialCutsCouple object(s).
  const G4MaterialCutsCouple* theG4MatCut = nullptr;
  if (theTestType < 2) {
      theG4MatCut = FakeG4Setup (theSecondaryProdCut, theTargetMaterialName, verbose);
  } else {
      FakeG4Setup (theSecondaryProdCut, verbose);
  }
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
  //     method for e- or e+ (regarding the global data, it could be any of e-: 0; e+: 1; or gamma: 2).
  G4HepEmRunManager*  runMgr = nullptr;
  if (theTestType !=1 ) {
    int theG4HepEmParticleIndx = theIsElectron ? 0 : 1; // e-: 0; e+: 1;
    runMgr = new G4HepEmRunManager ( true );
    runMgr->Initialize ( new G4HepEmCLHEPRandomEngine(G4Random::getTheEngine()), theG4HepEmParticleIndx );
  }

  //
  // --- Final state generation by using either the:
  //     - test type = 0: the G4HepEm interaction
  //     - test type = 1: the native Geant4 interaction model
  //     or if G4HepEm was built with CUDA support (-DG4HepEm_CUDA_BUILD=ON)
  //     - test type = 2: simple G4HepEm SB sampling table data host v.s device consistency check
  //
  switch (theTestType) {
      case 0:
          std::cout << " --- Final state test using the G4HepEm `" << theBremModelName << "` interaction model." << std::endl;
          G4HepEmSBTest(theG4MatCut, thePrimaryKinEnergy, theNumSamples, theNumHistBins, theBremModelName=="bremSB", theIsElectron);
          break;
      case 1:
          std::cout << " --- Final state test using the Geant4 `" << theBremModelName << "` interaction model." << std::endl;
          G4SBTest(theG4MatCut, thePrimaryKinEnergy, theNumSamples, theNumHistBins, theBremModelName=="bremSB", theIsElectron);
          break;
      default:
#ifdef G4HepEm_CUDA_BUILD
          std::cout << " --- Checking G4HepEmSBTableData host vs devide consistency." << std::endl;
          // make all G4HepEmData member available on the device (only if G4HepEm_CUDA_BUILD)
          CopyG4HepEmDataToGPU ( runMgr->GetHepEmData() );
          // invoke the SBTableDataTest
          if ( !TestSBTableData( runMgr->GetHepEmData() ) ) {
            std::cout << "     - G4HepEmSBTableData host vs device consistency test: FAILED....!!!  \n" << std::endl;
            return 1;
          } else if ( verbose > 0 ) {
            std::cout << "     - G4HepEmSBTableData host vs device consistency test: PASSED  \n" << std::endl;
          }
#else
          std::cerr << " *** Unknown test type: -t must be 0 or 1. " << std::endl;
#endif // G4HepEm_CUDA_BUILD
     }

  return 0;
}
