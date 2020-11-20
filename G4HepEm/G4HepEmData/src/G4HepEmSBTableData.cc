
#include "G4HepEmSBTableData.hh"

void AllocateSBTableData(struct G4HepEmSBTableData** theSBTableData, int numHepEmMatCuts, int numElemsInMC, int numSBData) {
  FreeSBTableData(theSBTableData);
  *theSBTableData = new G4HepEmSBTableData;
  (*theSBTableData)->fNumHepEmMatCuts             = numHepEmMatCuts;
  (*theSBTableData)->fGammaCutIndxStartIndexPerMC = new int[numHepEmMatCuts];
  for (int i=0; i<numHepEmMatCuts; ++i) {
    (*theSBTableData)->fGammaCutIndxStartIndexPerMC[i] = -1;
  }
  (*theSBTableData)->fGammaCutIndices             = new int[numElemsInMC];
  for (int i=0; i<numElemsInMC; ++i) {
    (*theSBTableData)->fGammaCutIndices[i] = -1;
  }
  //
  (*theSBTableData)->fNumSBTableData              = numSBData;
  (*theSBTableData)->fSBTableData                 = new double[numSBData];
}


void FreeSBTableData(struct G4HepEmSBTableData** theSBTableData) {
  if (*theSBTableData) {
    if ((*theSBTableData)->fGammaCutIndxStartIndexPerMC) {
      delete[] (*theSBTableData)->fGammaCutIndxStartIndexPerMC;
      (*theSBTableData)->fGammaCutIndxStartIndexPerMC = nullptr;
    }
    if ((*theSBTableData)->fGammaCutIndices) {
      delete[] (*theSBTableData)->fGammaCutIndices;
      (*theSBTableData)->fGammaCutIndices = nullptr;
    }
    if ((*theSBTableData)->fSBTableData) {
      delete[] (*theSBTableData)->fSBTableData;
      (*theSBTableData)->fSBTableData = nullptr;
    }
    delete (*theSBTableData);
    *theSBTableData = nullptr;
  }
} 
