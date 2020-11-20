
#include "G4HepEmMaterialData.hh"
#include <iostream>

// Allocates (the only one) G4HepEmMaterialData structure
void AllocateMaterialData(struct G4HepEmMaterialData** theMatData,  int numG4Mat, int numUsedG4Mat) {
  // clean away the previous (if any)
  FreeMaterialData ( theMatData );
  *theMatData = new G4HepEmMaterialData;
  (*theMatData)->fNumG4Material             = numG4Mat;
  (*theMatData)->fNumMaterialData           = numUsedG4Mat;
  (*theMatData)->fG4MatIndexToHepEmMatIndex = new int[numG4Mat];
  (*theMatData)->fMaterialData              = new G4HepEmMatData[numUsedG4Mat];
  // init G4Mat index to HepEmMat index translator to -1 (i.e. to `not used in the cur. geom.`)
  for ( int i=0; i<numG4Mat; ++i ) {
    (*theMatData)->fG4MatIndexToHepEmMatIndex[i] = -1;
  }
}


// Clears (the only one) G4HepEmMaterialData structure and resets its ptr to null 
void FreeMaterialData (struct G4HepEmMaterialData** theMatData) {
  if ( *theMatData ) {
    if ( (*theMatData)->fG4MatIndexToHepEmMatIndex ) {
      delete[] (*theMatData)->fG4MatIndexToHepEmMatIndex;
    }
    if ( (*theMatData)->fMaterialData ) {
      for (int imd=0; imd<(*theMatData)->fNumMaterialData; ++imd) {
        delete[] (*theMatData)->fMaterialData[imd].fNumOfAtomsPerVolumeVect;
        delete[] (*theMatData)->fMaterialData[imd].fElementVect; 
      }
      delete[] (*theMatData)->fMaterialData;
    }
    delete *theMatData;
    *theMatData = nullptr;
  }
}

