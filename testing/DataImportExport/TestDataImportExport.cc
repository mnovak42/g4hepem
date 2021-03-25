// local (and TestUtils) includes
#include "TestUtils/G4SetUp.hh"
#include "SimpleFakeG4Setup.h"

#include "G4HepEmDataJsonIO.hh"
#include "G4HepEmDataComparison.h"

// G4 includes
#include "globals.hh"
#include "G4SystemOfUnits.hh"
#include "Randomize.hh"
#include "G4ProductionCutsTable.hh"

// G4HepEm includes
#include "G4HepEmRunManager.hh"
#include "G4HepEmData.hh"
#include "G4HepEmCLHEPRandomEngine.hh"

int main(int argc, char* argv[])
{
  const std::string usage = "Usage: TestDataImportExport full|simple";
  std::string mode = "full";

  if (argc > 1) {
    if (argc != 2) {
      std::cerr << usage << std::endl;
      return 1;
    }

    mode = argv[1];

    if(mode != "full" && mode != "simple") {
      std::cerr << usage << std::endl;
      return 1;
    }
  }


  // Only output is the data file
  const G4String baseFilename = "G4HepEMTestDataImportExport_" + mode;
  const G4String g4hepemFile = baseFilename + ".json";

  // --- Set up a fake G4 geometry with including all pre-defined NIST materials
  //     to produce the G4MaterialCutsCouple objects.
  //
  // secondary production threshold in length
  const G4double secProdThreshold = 0.7 * mm;
  if (mode == "full") {
    FakeG4Setup(secProdThreshold);
  }
  else {
    SimpleFakeG4Setup(secProdThreshold);
  }

  // Dump cuts couples to check
  G4ProductionCutsTable::GetProductionCutsTable()->DumpCouples();

  // Construct the G4HepEmRunManager, which will fill the data structures
  // on calls to Initialize
  auto* runMgr    = new G4HepEmRunManager(true);
  auto* rngEngine = new G4HepEmCLHEPRandomEngine(G4Random::getTheEngine());
  runMgr->Initialize(rngEngine, 0);
  runMgr->Initialize(rngEngine, 1);
  runMgr->Initialize(rngEngine, 2);

  // Serialize to file
  G4HepEmData* outData = runMgr->GetHepEmData();
  {
    std::ofstream jsonOS{ g4hepemFile.c_str() };
    std::cout << "Serializing to " << g4hepemFile << "... " << std::flush;
    if(!G4HepEmDataToJson(jsonOS, outData))
    {
      std::cerr << "Failed to write G4HepEMData to " << g4hepemFile
                << std::endl;
      jsonOS.close();
      return 1;
    }
  }
  std::cout << "done" << std::endl;

  // Deserialize to check round trip
  std::cout << "Deserializing from " << g4hepemFile << "... " << std::flush;

  std::ifstream jsonIS{ g4hepemFile.c_str() };
  G4HepEmData* inData = G4HepEmDataFromJson(jsonIS);
  if(inData == nullptr)
  {
    std::cerr << "Failed to read G4HepEmData from " << g4hepemFile << std::endl;
    return 1;
  }
  std::cout << "done" << std::endl;

  // Validate that outData == inData
  std::cout << "Validating round-tripped G4HepEmData objects are numerically equal... ";
  if(*outData != *inData)
  {
    std::cerr << "Roundtripped G4HepEMData to->file->from instances are not "
                 "numerically identical!"
              << std::endl;
    FreeG4HepEmData(inData);
    return 1;
  }
  std::cout << "done" << std::endl;

  FreeG4HepEmData(inData);

  return 0;
}