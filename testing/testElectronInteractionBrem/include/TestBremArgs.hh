
#ifndef TESTBREMARGS_HH
#define TESTBREMARGS_HH

#include <string>

#include <getopt.h>

//
// Input argument of the `testBrem` model level test application
//


//
// simple structure to store the `testBrem` input arguments
struct BremArgs {
  std::string fParticleName;   // primary particle is electron
  std::string fMaterialName;   // material is lead
  std::string fBremModelName;  // name of the bremsstrahlung model to test
  int         fTestType;       // type of test (HepEm, G4 or GPU)
  int         fNumHistBins;    // number of histogram bins between min/max values
  double      fNumSamples;     // number of required final state samples
  double      fPrimaryEnergy;  // primary particle energy in [GeV]
  double      fProdCutValue;   // by default in length and internal units i.e. [cm]

  BremArgs():
  fParticleName("e-"),
  fMaterialName("G4_Pb"),
  fBremModelName("bremSB"),
  fTestType(0),
  fNumHistBins(100),
  fNumSamples(1.0E+7),
  fPrimaryEnergy(234.56),
  fProdCutValue(0.7) {}
};

static struct option options[] = {
    {"test-type         (G4HepEm`, `G4` or `GPU(data)`)    - default: HepEm",  required_argument, 0, 't'},
    {"particle-name     (`e-` or `e+`)                     - default: e-",     required_argument, 0, 'p'},
    {"material-name     (G4-NIST mat. name,`G4_` prefix)   - default: G4_Pb",  required_argument, 0, 'm'},
    {"primary-energy    (kinetic energy in [MeV])          - default: 234.56", required_argument, 0, 'e'},
    {"number-of-samples (number of required samples)       - default: 1.e+7",  required_argument, 0, 'f'},
    {"number-of-bins    (number of bins in the histograms) - default: 100",    required_argument, 0, 'n'},
    {"model-name        (`bremS` or `bremRel`)             - default: bremSB", required_argument, 0, 'b'},
    {"cut-vale          (secondary prod. thresh. in [mm])  - default: 0.7",    required_argument, 0, 'c'},
    {"help", no_argument, 0, 'h'},
    {0, 0, 0, 0}};

void GetBremArgs(int argc, char *argv[], struct BremArgs& args);
void GetBremArgsHelp();


#endif //  TESTBREMARGS_HH
