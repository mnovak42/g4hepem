#include "MockG4.h"
#include "G4GDMLParser.hh"
#include "G4ProductionCutsTable.hh"
#include "G4Version.hh"

#include <cstdio>

int main(int argc, char* argv[])
{
  // Only outputs are the data file(s)
  // Separate for now, but will want to unify/connect
  const G4String baseFilename = "G4HepEMTestDataImportExport";
  const G4String gdmlFile     = baseFilename + ".gdml";
  const G4String g4hepemFile  = baseFilename + ".g4hepem";

  // Build mock geant4 setup
  // - Should create geometry, regions and cuts
  G4PVPlacement* world = MockG4();

  // Dump cuts couples to check
  G4ProductionCutsTable::GetProductionCutsTable()->DumpCouples();

  // Persist data
  // Add export of regions and energy cuts to see what these
  // do and how to use. Remove pointer from exported names for
  // now to aid reabability (we know we won't have duplicated names)
  G4GDMLParser gdmlParser;
  gdmlParser.SetAddPointerToName(false);
// Only for Geant4 10.7!
#if G4VERSION > 1069
  gdmlParser.SetOutputFileOverwrite(true);
#else
  std::remove(gdmlFile.c_str());
#endif
  gdmlParser.SetRegionExport(true);
  gdmlParser.SetEnergyCutsExport(true);
  gdmlParser.Write(gdmlFile, world, false);

  return 0;
}