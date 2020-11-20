
#include "G4HepEmElementData.hh"

void AllocateElementData(struct G4HepEmElementData** theElementData) {
  // clean away the previous (if any)
  FreeElementData ( theElementData );
  *theElementData   = new G4HepEmElementData;
  int maxZetPlusOne = 121;
  (*theElementData)->fMaxZet      = maxZetPlusOne-1;
  (*theElementData)->fElementData = new G4HepEmElemData[maxZetPlusOne];
  // init the all Z to -1 to indicate that it has not been set
  for (int ie=0; ie<(*theElementData)->fMaxZet; ++ie) {
    (*theElementData)->fElementData[ie].fZet = -1.0;
  }
}


void FreeElementData(struct G4HepEmElementData** theElementData) {
  if ( *theElementData ) {
    if ( (*theElementData)->fElementData ) {
      delete[] (*theElementData)->fElementData;
    }
    delete *theElementData;
    *theElementData = nullptr;
  }
}
