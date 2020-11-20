
#include "G4HepEmMatCutData.hh"
#include <iostream>

//#include <cstdlib>

// Allocates (the only one) G4HepEmMatCutData structure
void AllocateMatCutData(struct G4HepEmMatCutData** theMatCutData, int numG4MatCuts, int numUsedG4MatCuts) {
  // clean away the previous (if any)
  FreeMatCutData ( theMatCutData );
  *theMatCutData = new G4HepEmMatCutData;
  (*theMatCutData)->fNumG4MatCuts            = numG4MatCuts;
  (*theMatCutData)->fNumMatCutData           = numUsedG4MatCuts;
  (*theMatCutData)->fG4MCIndexToHepEmMCIndex = new int[numG4MatCuts];
  (*theMatCutData)->fMatCutData              = new G4HepEmMCCData[numUsedG4MatCuts];
  // init G4MC index to HepEmMC index translator to -1 (i.e. to `not used in the cur. geom.`)
  for ( int i=0; i<numG4MatCuts; ++i ) {
    (*theMatCutData)->fG4MCIndexToHepEmMCIndex[i] = -1;
  }
}


// Clears (the only one) G4HepEmMatCutData structure and resets its ptr to null 
void FreeMatCutData (struct G4HepEmMatCutData** theMatCutData) {
  if ( *theMatCutData ) {
    if ( (*theMatCutData)->fG4MCIndexToHepEmMCIndex ) {
      delete[] (*theMatCutData)->fG4MCIndexToHepEmMCIndex;
    }
    if ( (*theMatCutData)->fMatCutData ) {
      delete[] (*theMatCutData)->fMatCutData;
    }
    delete *theMatCutData;
    *theMatCutData = nullptr;
  }
}

