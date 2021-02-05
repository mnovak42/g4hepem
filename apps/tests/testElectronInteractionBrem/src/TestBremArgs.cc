
#include "TestBremArgs.hh"

#include <iostream>
#include <iomanip>
#include <cstdio>

#include <err.h>


// get test application input arguments
void GetBremArgs(int argc, char *argv[], struct BremArgs& args) {
  while (true) {
    int c, optidx = 0;
    c = getopt_long(argc, argv, "t:p:m:e:f:n:b:c:", options, &optidx);
    if (c == -1) break;
    switch (c) {
    case 0:
      c = options[optidx].val;
    /* fall through */
    case 't':
      args.fTestType   = (int)strtof(optarg, NULL);
      if (args.fTestType < 0 || args.fTestType > 2 ) {
        GetBremArgsHelp();
        errx(1, "test type -t must be 0(HepEM), 1(G4) or 2(GPU-data)");
      }
      break;
    case 'p':
      args.fParticleName = optarg;
      if (!(args.fParticleName == "e-" || args.fParticleName == "e+")) {
        GetBremArgsHelp();
        errx(1, "unknown particle name");
      }
      break;
    case 'm':
      args.fMaterialName = optarg;
      break;
    case 'e':
      args.fPrimaryEnergy = strtod(optarg, NULL);
      if (args.fPrimaryEnergy <= 0) {
        GetBremArgsHelp();
        errx(1, "primary particle energy must be positive");
      }
      break;
    case 'f':
      args.fNumSamples = strtod(optarg, NULL);
      if (args.fNumSamples <= 0) {
        GetBremArgsHelp();
        errx(1, "number of final state samples must be positive");
      }
      break;
    case 'n':
      args.fNumHistBins = (int)strtof(optarg, NULL);
      if (args.fNumHistBins <= 0) {
        GetBremArgsHelp();
        errx(1, "number of histogram bins must be positive");
      }
      break;
    case 'b':
      args.fBremModelName = optarg;
      if (!(args.fBremModelName == "bremSB" || args.fBremModelName == "bremRel")) {
        GetBremArgsHelp();
        errx(1, "unknown bremsstrahlung model name");
      }
      break;
    case 'c':
      args.fProdCutValue = strtod(optarg, NULL);
      if (args.fProdCutValue <= 0) {
        errx(1, "production cut value must be positive");
        GetBremArgsHelp();
      }
      break;
    case 'h':
      GetBremArgsHelp();
      return;
    default:
      GetBremArgsHelp();
      errx(1, "unknown option %c", c);
    }
  }
}


void GetBremArgsHelp() {
  std::cout << "\n " << std::setw(90) << std::setfill('=') << "" << std::setfill(' ') << std::endl;
  std::cout << "  Model-level test for testing e-/e+ bremsstrahlung photon emission intercation models."
            << std::endl;
  std::cout << "\n  Usage: testBrem [OPTIONS] \n" << std::endl;
  for (int i = 0; options[i].name != NULL; i++) {
    printf("\t-%c  --%s\n", options[i].val, options[i].name);
  }
  std::cout << "\n " << std::setw(90) << std::setfill('=') << "" << std::setfill(' ') << std::endl;
}
